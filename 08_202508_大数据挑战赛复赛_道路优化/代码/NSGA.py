import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import io
import random
import warnings
warnings.filterwarnings('ignore')
from deap import base, creator, tools, algorithms

def safe_redirect():
    """安全处理终端编码，避免强制重定向导致崩溃"""
    try:
        current_enc = sys.stdout.encoding.lower() if sys.stdout.encoding else 'unknown'
        if current_enc not in ['utf-8', 'utf8']:
            # 行缓冲模式：输出立即刷新，减少编码冲突
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                line_buffering=True
            )
            print(f"✅ 终端编码重定向完成（原编码：{current_enc} → 目标编码：utf-8）")
        else:
            print(f"✅ 终端已支持UTF-8编码（当前编码：{current_enc}），无需重定向")
    except Exception as e:
        print(f"⚠️  编码重定向失败（不影响核心功能），错误：{str(e)[:50]}")

# -------------------------- 1. 配置参数（指定文件定义） --------------------------
# 输入数据：预测结果 + 初步预处理数据
INPUT_PRED_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/任务B预测结果/维护需求预测结果.csv"
INPUT_PREPROCESSED_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/processed_road_data.csv"
OUTPUT_OPT_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/任务B优化结果"
os.makedirs(OUTPUT_OPT_PATH, exist_ok=True)

# 新增：定义ID列名（与论文初步预处理一致，B2025070938351.pdf4.4.1节）
ID_COL = "ID"  # Segment ID经初步预处理后简化为"ID"

# 业务参数（第二问.pdf1-25/1-31/1-36节）
TOTAL_BUDGET = 1000000    # 总预算100万元
TOTAL_H_MANPOWER = 500    # 总人员工时500小时/月
TOTAL_M_EQUIP = 300       # 总设备台时300台时/月
RAINFALL_THRESHOLD = 1500 # 多雨路段阈值1500mm
MAX_RAINY_MAINTENANCE = 15# 雨季前最多维护15个多雨路段
C_BASE = 10000            # 基础维护成本1万元/段
AADT_MAX = 20000          # 最大AADT（归一化用）
MAX_YEARS = 20            # 最大维护间隔年数（归一化用）
# 道路权重（第二问.pdf1-31节）
ROAD_WEIGHT = {1: 0, 2: 0.15, 3: 0.3}  # RT=1(支路)=0、RT=2(次干道)=0.15、RT=3(主干道)=0.3
ROAD_H_MAP = {1: 5, 2: 8, 3: 10}       # 人员工时：支路=5、次干道=8、主干道=10

# -------------------------- 2. 核心参数计算（基于初步预处理数据） --------------------------
def calculate_optimization_params(pred_df, preprocessed_df):
    """
    利用论文初步预处理的RT/AT/YSM等列，计算优化所需参数（第二问.pdf1-27/1-31/1-36节）
    1. 路段优先级P_i；2. 维护成本C_i；3. 交通影响时间T_i
    """
    print("="*50)
    print("步骤1：计算优化核心参数（第二问.pdf1-27节）")
    print("="*50)
    
    # 合并数据（预测结果+初步预处理特征）
    merge_cols = [ID_COL, "RT", "AT", "PCI", "R", "AADT", "AR", "YSM"]
    df = pd.merge(pred_df, preprocessed_df[merge_cols], on=ID_COL, how="inner")
    
    # 1. 路段优先级P_i（第二问.pdf公式：0.3*y_hat_i + 0.25*(1-PCI/100) + ...）
    df["P_i"] = (0.3 * df["y_hat_i"] + 
                 0.25 * (1 - df["PCI"]/100) + 
                 0.2 * (df["R"]/30) + 
                 0.15 * (df["AADT"]/AADT_MAX) + 
                 0.1 * (df["YSM"]/MAX_YEARS))
    df["P_i"] = df["P_i"].clip(0, 1)  # 取值0~1（论文要求）
    
    # 2. 维护成本C_i（第二问.pdf公式：C_base*(1+0.3*Road_Weight+0.2*Asphalt_Weight+0.1*R/30)）
    # Asphalt_Weight：AT=1(混凝土)=0、AT=2(沥青)=0.1（第二问.pdf1-9节）
    df["Road_Weight"] = df["RT"].map(ROAD_WEIGHT)
    df["Asphalt_Weight"] = df["AT"].map({1: 0, 2: 0.1})
    df["C_i"] = C_BASE * (1 + 0.3*df["Road_Weight"] + 0.2*df["Asphalt_Weight"] + 0.1*(df["R"]/30))
    
    # 3. 交通影响时间T_i（第二问.pdf公式：2*(AADT/AADT_MAX)）
    df["T_i"] = 2 * (df["AADT"]/AADT_MAX)
    
    # 4. 约束参数（人员工时H_i、设备台时M_i、多雨标识）
    df["H_i"] = df["RT"].map(ROAD_H_MAP)  # 基于初步预处理的RT
    df["M_i"] = 1.5  # 每路段需1.5设备台时（第二问.pdf1-50节）
    df["Is_Rainy"] = (df["AR"] > RAINFALL_THRESHOLD).astype(int)
    
    # 保留核心参数列
    core_params = [ID_COL, "y_hat_i", "P_i", "C_i", "T_i", "H_i", "M_i", "Is_Rainy", "RT", "PCI"]
    print(f"✅ 核心参数计算完成：{df.shape[0]}个路段")
    return df[core_params]

# -------------------------- 3. NSGA-II多目标优化（第二问.pdf1-42节） --------------------------
def nsga2_optimize(param_df):
    """
    目标函数：最大化F1（维护价值）、最小化F2（成本）、最小化F3（交通影响）
    约束条件：预算、人员、设备、气候（第二问.pdf1-48节）
    修复：初始种群预先评估适应度，确保所有个体有完整的3目标适应度
    """
    print("\n" + "="*50)
    print("步骤2：NSGA-II多目标优化（第二问.pdf1-42节）")
    print("="*50)
    
    n_segs = len(param_df)
    # 提取参数数组（与原代码一致）
    P = param_df["P_i"].values
    C = param_df["C_i"].values
    T = param_df["T_i"].values
    H = param_df["H_i"].values
    M = param_df["M_i"].values
    Is_Rainy = param_df["Is_Rainy"].values
    
    # DEAP框架初始化（与原代码一致）
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))  # 3目标：max(F1), min(F2,F3)
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_segs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 评估函数（与原代码一致，确保返回3元素元组）
    def evaluate(ind):
        x = np.array(ind)
        F1 = np.sum(P * x)
        F2 = np.sum(C * x)
        F3 = np.sum(T * x)
        constraints = [
            F2 <= TOTAL_BUDGET,
            np.sum(H * x) <= TOTAL_H_MANPOWER,
            np.sum(M * x) <= TOTAL_M_EQUIP,
            np.sum(x * Is_Rainy) <= MAX_RAINY_MAINTENANCE
        ]
        if not all(constraints):
            return (F1 - 1.5, F2 + 3e5, F3 + 30)  # 强制3元素元组
        return (F1, F2, F3)  # 强制3元素元组
    
    # 遗传操作（与原代码一致）
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)
    
    # -------------------------- 关键修复：初始种群预先评估适应度 --------------------------
    random.seed(42)
    pop = toolbox.population(n=80)  # 创建初始种群
    # 新增：评估初始种群所有个体，确保每个个体都有3目标适应度
    print("  初始化种群并评估适应度...")
    initial_fitnesses = list(toolbox.map(toolbox.evaluate, pop))  # 批量评估
    for ind, fit in zip(pop, initial_fitnesses):
        ind.fitness.values = fit  # 显式赋值3元素适应度元组
        # 验证：确保适应度长度为3（可选，用于调试）
        if len(ind.fitness.values) != 3:
            print(f"⚠️  初始个体适应度长度异常：{len(ind.fitness.values)}，需检查评估函数")
            raise SystemExit(1)
    print("  初始种群适应度评估完成")
    
    # 迭代优化（与原代码一致，仅补充打印）
    n_gen = 40
    for gen in range(n_gen):
        # 1. 生成后代（交叉+变异）
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.1)
        # 2. 评估后代适应度（确保所有后代有3目标值）
        offspring_fitnesses = list(toolbox.map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, offspring_fitnesses):
            ind.fitness.values = fit
        # 3. 选择（合并父代与后代，选择最优个体）
        pop = toolbox.select(offspring + pop, k=len(pop))
        
        # 补充：打印迭代进度（可选，便于调试）
        if (gen + 1) % 10 == 0:
            print(f"  迭代{gen+1}/40完成（当前种群个体数：{len(pop)}）")
    
    # 5. 提取Pareto最优解集（与原代码一致）
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pareto_results = []
    for ind in pareto_front:
        x = np.array(ind)
        F1 = np.sum(P * x)
        F2 = np.sum(C * x)
        F3 = np.sum(T * x)
        pareto_results.append({
            "F1_维护价值": round(F1, 4),
            "F2_总成本(元)": round(F2, 2),
            "F3_总交通影响(小时)": round(F3, 2),
            "维护路段数": int(np.sum(x)),
            "决策变量x_i": x.tolist()
        })
    pareto_df = pd.DataFrame(pareto_results)
    pareto_path = os.path.join(OUTPUT_OPT_PATH, "Pareto最优解集.csv")
    pareto_df.to_csv(pareto_path, index=False, encoding="utf-8-sig")
    print(f"✅ Pareto最优解集保存至：{pareto_path}（{len(pareto_df)}个解）")
    
    # -------------------------- 关键修复：推荐方案筛选逻辑 --------------------------
    print("  筛选推荐方案（第二问.pdf1-66节：基于业务偏好选择）...")
    # 修复1：放宽筛选条件为“总成本≤总预算”（符合第二问.pdf1-48节预算约束）
    valid_solutions = pareto_df[pareto_df["F2_总成本(元)"] <= TOTAL_BUDGET]
    
    # 修复2：处理无有效解的极端情况
    if valid_solutions.empty:
        print(f"  ⚠️  无符合预算（{TOTAL_BUDGET}元）的解，选择成本最接近预算的解")
        # 计算每个解与预算的差值，选择差值最小的解
        pareto_df["成本差值"] = np.abs(pareto_df["F2_总成本(元)"] - TOTAL_BUDGET)
        recommended = pareto_df.sort_values("成本差值").iloc[0]
    else:
        # 正常情况：选择F1（维护价值）最大的解（第二问.pdf1-46节目标函数优先级）
        recommended = valid_solutions.sort_values("F1_维护价值", ascending=False).iloc[0]
    
    # 提取推荐方案的维护路段
    recommended_x = np.array(recommended["决策变量x_i"])
    recommend_segs = param_df[recommended_x == 1][ID_COL].tolist()
    
    # 保存推荐方案（与原代码一致）
    recommend_df = pd.DataFrame({
        "推荐方案指标": ["维护价值(F1)", "总成本(F2)", "总交通影响(F3)", "维护路段数", "高优先级维护占比(%)"],
        "数值": [recommended["F1_维护价值"], f"{recommended['F2_总成本(元)']:.2f}", 
                f"{recommended['F3_总交通影响(小时)']:.2f}", recommended["维护路段数"],
                round(param_df[np.array(recommended["决策变量x_i"]) == 1]["P_i"].ge(0.7).mean()*100, 2)]
    })
    recommend_path = os.path.join(OUTPUT_OPT_PATH, "推荐优化方案.csv")
    recommend_df.to_csv(recommend_path, index=False, encoding="utf-8-sig")
    print(f"✅ 推荐方案保存至：{recommend_path}（维护{len(recommend_segs)}个路段）")
    
    return pareto_df, recommended, recommend_segs

# -------------------------- 4. 优化效果评估（第二问.pdf1-55节） --------------------------
def evaluate_optimization(param_df, recommended_x, recommend_segs):
    x = np.array(recommended_x)
    # 1. 维护覆盖率（TP_maintained/TP_total，第二问.pdf1-55节）
    TP_total = param_df[param_df["y_hat_i"] == 1].shape[0]
    TP_maintained = param_df[(param_df["y_hat_i"] == 1) & (x == 1)].shape[0]
    coverage = (TP_maintained / TP_total * 100) if TP_total > 0 else 0.0
    
    # 2. 资源效率（预算/人员利用率）
    actual_cost = np.sum(param_df["C_i"] * x)
    budget_util = (actual_cost / TOTAL_BUDGET * 100)
    actual_h = np.sum(param_df["H_i"] * x)
    h_util = (actual_h / TOTAL_H_MANPOWER * 100)
    
    # 3. 平均PCI提升率（第二问.pdf1-55节：假设维护后PCI=90）
    pci_before = param_df[param_df[ID_COL].isin(recommend_segs)]["PCI"].mean()
    pci_increase = ((90 - pci_before) / pci_before * 100) if pci_before > 0 else 0.0
    
    # 4. 平均交通影响指数
    avg_t = np.sum(param_df["T_i"] * x) / len(recommend_segs) if len(recommend_segs) > 0 else 0.0
    
    # 5. 高优先级维护占比（P_i≥0.7）
    high_p_ratio = (param_df[(param_df["P_i"] >= 0.7) & (x == 1)].shape[0] / len(recommend_segs) * 100) if len(recommend_segs) > 0 else 0.0
    
    # 整理评估结果
    eval_df = pd.DataFrame({
        "评估维度": ["风险覆盖", "资源效率", "资源效率", "维护效果", "交通影响", "优先级匹配"],
        "指标名称": ["维护覆盖率", "预算利用率", "人员工时利用率", "平均PCI提升率", "平均交通影响指数", "高优先级维护占比"],
        "数值(%)": [round(coverage, 2), round(budget_util, 2), round(h_util, 2), 
                  round(pci_increase, 2), round(avg_t, 2), round(high_p_ratio, 2)],
        "论文优秀标准": ["≥85%", "95%~100%", "90%~100%", "≥20%", "≤2小时", "≥80%"]
    })
    # 修正交通影响指数单位显示
    eval_df.loc[eval_df["指标名称"] == "平均交通影响指数", "数值(%)_单位"] = "小时"
    
    # 保存评估结果
    eval_path = os.path.join(OUTPUT_OPT_PATH, "优化效果评估.csv")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    
    # 打印结果
    print("\n" + "="*50)
    print("优化效果评估（第二问.pdf1-55节）")
    print("="*50)
    print(eval_df.to_string(index=False, columns=["评估维度", "指标名称", "数值(%)", "论文优秀标准"]))
    return eval_df

# -------------------------- 5. 优化主流程 --------------------------
if __name__ == "__main__":

    safe_redirect()

    # 1. 加载数据（预测结果+初步预处理数据）
    pred_df = pd.read_csv(INPUT_PRED_PATH)
    preprocessed_df = pd.read_csv(INPUT_PREPROCESSED_PATH)
    print(f"✅ 数据加载成功：预测结果{pred_df.shape[0]}行，初步预处理数据{preprocessed_df.shape[0]}行")
    
    # 2. 计算优化核心参数（基于初步预处理的特征）
    param_df = calculate_optimization_params(pred_df, preprocessed_df)
    
    # 3. NSGA-II优化
    pareto_df, recommended, recommend_segs = nsga2_optimize(param_df)
    
    # 4. 评估优化效果
    evaluate_optimization(param_df, recommended["决策变量x_i"], recommend_segs)
    
    print(f"\n✅ 资源分配优化代码执行完成！输出路径：{OUTPUT_OPT_PATH}")