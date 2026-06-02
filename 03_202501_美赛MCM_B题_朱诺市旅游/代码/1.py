import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# ======================== 参数设置 ========================
# 调整这些参数以探索不同场景
N_GENERATIONS = 200  # 迭代次数
POPULATION_SIZE = 100  # 种群大小

# 成本收入模型参数
k = 0.8  # 游客消费与收入的比例系数
t = 0.1  # 提高的税率
C1 = 1e6  # 基础设施建设成本
CE = 5e5  # 环境治理成本
CF = 2e5  # 当地民众反馈成本

# 环境影响评估模型参数
alpha = 0.02  # 环境代价系数
a = 1e6  # 环境承载能力常数
b = 0.5  # 环境承载能力衰减系数

# 社区满意度平衡模型参数
w1, w2, w3 = 0.4, 0.3, 0.3  # 满意度权重
c = 0.05  # 就业机会与游客人数关系系数
J_max = 10000  # 最大就业岗位数
N_max = 1e6  # 最大环境承载游客数

# ======================== 输入数据 ========================
# 时间维度数据
years = [2021, 2022, 2023]
visitors = [1305000, 1167000, 1650000]

# 旅游业影响评价数据
negative_impact = [0.08, 0.07, 0.11]
positive_impact = [0.36, 0.35, 0.31]
mixed_impact = [0.33, 0.49, 0.38]
no_impact = [0.2, 0.16, 0.11]

# 企业基金相关数据
water_revenue = [5853331, 6144965, 6271775]
water_expense = [5886606, 6332522, 6456579]
wastewater_revenue = [13465346, 14105534, 14525481]
wastewater_expense = [10841401, 12835462, 13292894]

# 港口发展数据
port_revenue = [2446364, 1556973, 8265864]
port_expense = [7600, 7600, 7600]

# 海运乘客费
ocean_passenger_revenue = [8570, 2641085, 6988700]
ocean_passenger_expense = [7500, 7500, 7500]

# 垃圾处理数据
commercial_waste = {"recyclable": 2343, "compostable": 3823, "special_recycling": 1959, "reusable": 1316, "non_recyclable": 1946, "total": 11388}
residential_waste = {"recyclable": 1361, "compostable": 2942, "special_recycling": 1469, "reusable": 169, "non_recyclable": 2045, "total": 7985}

# 人口统计数据
population = [31773, 31773, 31773]
per_capita_income = [74339, 74339, 74339]

# ======================== 旅游业发展问题决策变量范围 ========================
N_MIN, N_MAX = 1e5, 2e6  # 游客人数范围

# ======================== 自定义问题 ========================
class TourismProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1,  # 决策变量个数 (游客人数)
                         n_obj=3,  # 目标个数 (收益、环境影响、满意度)
                         n_constr=0,  # 约束个数
                         xl=np.array([N_MIN]),  # 决策变量下界
                         xu=np.array([N_MAX]))  # 决策变量上界

    def _evaluate(self, x, out, *args, **kwargs):
        N = x[:, 0]  # 决策变量 (游客人数)

        # 成本收入模型计算收益
        RT = k * N
        RTAX = t * RT
        P = (1 + t) * RT - C1 - CE - CF

        # 环境影响评估模型计算环境影响
        I_C = 0.001 * N  # 游客碳足迹影响
        I_E = 0.002 * N  # 能源消耗影响
        I_G = 0.0015 * N  # 垃圾处理影响
        I = alpha * (I_C + I_E + I_G)

        # 社区满意度平衡模型计算满意度
        J = c * N
        S1 = np.where(J <= J_max, J / J_max, 1)  # 就业满意度
        S3 = np.where(N <= N_max, 1 - N / N_max, 0)  # 生活质量满意度
        S2 = 0.8  # 假定收入满意度为常数
        S = w1 * S1 + w2 * S2 + w3 * S3

        # 将目标值返回
        out["F"] = np.column_stack([-P, I, -S])  # 最大化 P 和 S，所以取负值

# ======================== 优化算法设置 ========================
algorithm = NSGA2(pop_size=POPULATION_SIZE)
termination = get_termination("n_gen", N_GENERATIONS)

# ======================== 运行优化 ========================
problem = TourismProblem()
result = minimize(problem,
                  algorithm,
                  termination,
                  seed=42,
                  save_history=True,
                  verbose=True)

# ======================== 可视化与结果 ========================
# 绘制三目标的散点图
plot = Scatter(title="Pareto Front for Tourism Optimization")
plot.add(result.F, facecolor="red", edgecolor="black")
plot.show()

# 保存结果到表格文件
data = {
    "Solution": [f"Solution {i + 1}" for i in range(len(result.X))],
    "N (Tourists)": [x[0] for x in result.X],
    "Profit": [-f[0] for f in result.F],
    "Impact": [f[1] for f in result.F],
    "Satisfaction": [-f[2] for f in result.F]
}
results_df = pd.DataFrame(data)
results_df.to_csv("tourism_optimization_results.csv", index=False)

# 输出最佳解
best_index = np.argmax([-f[0] for f in result.F])  # 最大化收益作为选择标准
best_solution = result.X[best_index]
best_objectives = result.F[best_index]

print("\nBest Solution:")
print(f"N (Tourists): {best_solution[0]:,.0f}")
print(f"Profit: {-best_objectives[0]:,.2f}")
print(f"Impact: {best_objectives[1]:,.2f}")
print(f"Satisfaction: {-best_objectives[2]:.4f}")
