import numpy as np
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt

# ================== 全局配置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体配置
plt.rcParams['axes.unicode_minus'] = False     
t_first_landing = 1.5    # 首个运动员落地时间（s）
MIN_INTERVAL = 0.3       # 最小接触间隔约束（s）
PENALTY_FACTOR = 0.5     # 约束惩罚系数
SPRING_NUM = 20          # 总弹簧数量（根据实际调整）

# ================== 最优参数（来自灵敏度分析） ==================
OPT_POP_SIZE = 200       # 种群大小
OPT_CXPB = 0.8           # 交叉概率
OPT_MUTPB = 0.4          # 变异概率
OPT_SIGMA = 0.15         # 变异步长（高斯变异标准差）
OPT_TOURNSIZE = 2        # 锦标赛选择轮数

# ================== 动力学仿真（需替换为实际模型） ==================
def simulate_dynamics(t):
    """
    动力学仿真与损伤计算（伪代码）
    参数: t (array) - 5个运动员的起跳时间序列（已排序）
    返回: (D_max, stress_distribution, fatigue_life)
        D_max: 最大损伤值
        stress_distribution: 弹簧应力矩阵（SPRING_NUM x 7）
        fatigue_life: 各弹簧疲劳寿命（长度=SPRING_NUM）
    """
    # 示例逻辑：时间越分散，损伤越小（需替换为真实模型）
    time_spread = np.max(t) - np.min(t)
    D_max = 0.5 - 0.1 * time_spread  
    
    # 生成伪应力分布（0-150 MPa，SPRING_NUM个弹簧×7个时间点）
    stress_distribution = np.random.rand(SPRING_NUM, 7) * 150
    
    # 生成伪疲劳寿命（与损伤成反比，与应力负相关）
    fatigue_life = 10000 / (D_max + 0.03 * np.arange(1, SPRING_NUM+1))  
    return D_max, stress_distribution, fatigue_life

# ================== 遗传算法评估函数 ==================
def evaluate(individual):
    """评估个体适应度（目标：最小化损伤+惩罚）"""
    t = np.sort(individual)  # 确保时间有序
    penalty = 0.0
    
    # 约束1：所有时间不超过首个落地时间
    if np.any(t > t_first_landing):
        penalty += PENALTY_FACTOR * (np.max(t) - t_first_landing)  # 超时惩罚
    
    # 约束2：相邻时间间隔≥MIN_INTERVAL
    intervals = np.diff(t)  # 计算相邻间隔
    if np.min(intervals) < MIN_INTERVAL:
        penalty += PENALTY_FACTOR * (MIN_INTERVAL - np.min(intervals))  # 间隔不足惩罚
    
    # 调用动力学仿真计算损伤值
    D_max, _, _ = simulate_dynamics(t)
    
    # 适应度函数：损伤值 + 时间分散性项 + 惩罚项
    return (D_max + 0.1 * np.sum(intervals) + penalty * D_max,)

# ================== 遗传算法配置（使用最优参数） ==================
# 定义适应度（最小化问题）和个体类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 个体初始化：在[0, t_first_landing]范围内生成5个时间点
toolbox.register("attr_float", np.random.uniform, 0, t_first_landing)
toolbox.register("individual", tools.initRepeat, 
                 creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 带边界约束的高斯变异（使用最优sigma=0.15）
def bounded_mutate(individual, indpb):
    for i in range(len(individual)):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(0, OPT_SIGMA)  # 变异步长=0.15
            individual[i] = np.clip(individual[i], 0, t_first_landing)  # 限制时间范围
    return individual,

# 遗传操作（使用最优参数）
toolbox.register("mate", tools.cxBlend, alpha=0.3)       # 混合交叉
toolbox.register("mutate", bounded_mutate, indpb=OPT_MUTPB)  # 变异概率=0.4
toolbox.register("select", tools.selTournament, tournsize=OPT_TOURNSIZE)  # 锦标赛轮数=2
toolbox.register("evaluate", evaluate)  # 评估函数

# ================== 结果可视化函数 ==================
def plot_evolution_curve(logbook):
    """绘制遗传算法进化曲线"""
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    plt.figure(figsize=(10, 6))
    plt.plot(gen, min_fitness, "b-", linewidth=2)
    plt.xlabel("进化代数", fontsize=12)
    plt.ylabel("最小适应度（损伤+惩罚）", fontsize=12)
    plt.title("最优参数下遗传算法进化过程", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def plot_stress_comparison(baseline_stress, optimized_stress):
    """对比基准与优化后的弹簧应力分布"""
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(baseline_stress, cmap="hot", vmin=0, vmax=150)
    plt.title("基准应力分布 (MPa)")
    plt.colorbar(shrink=0.8)
    plt.xlabel("时间点")
    plt.ylabel("弹簧编号")
    
    plt.subplot(1, 2, 2)
    plt.imshow(optimized_stress, cmap="hot", vmin=0, vmax=150)
    plt.title("优化后应力分布 (MPa)")
    plt.colorbar(shrink=0.8)
    plt.xlabel("时间点")
    plt.ylabel("弹簧编号")
    plt.tight_layout()
    plt.show()

def plot_top80_life_improvement(baseline_life, optimized_life, baseline_stress):
    """绘制80%关键弹簧寿命提升对比图（按基准应力筛选）"""
    top_n = int(np.ceil(SPRING_NUM * 0.8))  # 80%关键弹簧数量
    sorted_indices = np.argsort(baseline_stress)[::-1]  # 按基准应力降序排序
    top_indices = sorted_indices[:top_n]                # 前80%关键弹簧
    
    baseline_top = baseline_life[top_indices]
    optimized_top = optimized_life[top_indices]
    labels = [f"弹簧{i+1}" for i in top_indices]         # 弹簧标签
    
    x = np.arange(top_n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(16, 7))
    
    rects1 = ax.bar(x - width/2, baseline_top, width, 
                    label="基准寿命", color="#4CAF50", edgecolor="k", alpha=0.8)
    rects2 = ax.bar(x + width/2, optimized_top, width, 
                    label="优化后寿命", color="#2196F3", edgecolor="k", alpha=0.8)
    
    ax.bar_label(rects1, padding=3, fmt="%d")  # 柱顶数值标签
    ax.bar_label(rects2, padding=3, fmt="%d")
    
    ax.set_ylabel("疲劳寿命（循环次数）", fontsize=12)
    ax.set_xlabel(f"关键弹簧（基准应力前{top_n}/{SPRING_NUM}个，占80%）", fontsize=12)
    ax.set_title("80%关键弹簧疲劳寿命提升对比", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.1 * max(np.max(baseline_top), np.max(optimized_top)))
    
    plt.tight_layout()
    plt.show()

# ================== 主程序（运行最终优化） ==================
if __name__ == "__main__":
    # 初始化种群（使用最优种群大小=200）
    population = toolbox.population(n=OPT_POP_SIZE)
    
    # 定义统计指标（记录最小适应度）
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    # 运行遗传算法（使用最优交叉概率=0.8，变异概率=0.4）
    print("正在运行遗传算法优化...")
    result, logbook = algorithms.eaSimple(
        population, toolbox, 
        cxpb=OPT_CXPB,    # 交叉概率=0.8
        mutpb=OPT_MUTPB,  # 变异概率=0.4
        ngen=200,         # 进化代数=200
        stats=stats, 
        verbose=True
    )
    
    # 提取最优解
    best_ind = tools.selBest(result, k=1)[0]
    best_times = np.sort(best_ind)
    best_fitness = logbook.select("min")[-1]
    
    # 输出结果
    print("\n========== 最终优化结果 ==========")
    print(f"最优起跳时间序列（排序后）: {best_times.round(2)} s")
    print(f"最优适应度（损伤+惩罚）: {best_fitness:.4f}")
    
    # 获取仿真数据（基准方案 vs 优化方案）
    _, baseline_stress, baseline_life = simulate_dynamics([0.2, 0.5, 0.8, 1.1, 1.4])  # 基准时间
    _, optimized_stress, optimized_life = simulate_dynamics(best_times)             # 优化时间
    baseline_stress_max = np.max(baseline_stress, axis=1)  # 各弹簧最大应力（用于筛选关键弹簧）
    
    # 绘制可视化图表
    plot_evolution_curve(logbook)                # 进化曲线
    plot_stress_comparison(baseline_stress, optimized_stress)  # 应力分布对比
    plot_top80_life_improvement(baseline_life, optimized_life, baseline_stress_max)  # 80%关键弹簧寿命对比