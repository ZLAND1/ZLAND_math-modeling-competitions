import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ================== 数据模拟 ==================
# 模拟基础数据（基于PDF中2023年基线数据）
np.random.seed(42)

# 时间范围（2023年）
days = 365
months = 12

# 基础参数
base_params = {
    'x_ship': 3000,      # 邮轮日均量（优化变量初始值）
    'x_land': 1200,      # 独立游客日均量（优化变量初始值）
    'eta_emission': 0.10,# 减排效率
    'Q_max': 1200,       # 动态环境承载力
    'housing_cost_ratio': 0.25,  # 住房成本占比
    'eco_repair_ratio': 0.40,    # 生态修复支出占比
    'tax_ship': 78,      # 邮轮税基价（美元/人）
    'tax_land': 18       # 陆地游客税基价（美元/人）
}

# 生成月度波动数据（旺季6-8月）
seasonal_factor = np.where(np.arange(1,13) >=6, 1.2, 1.0)  # 旺季系数

# ================== 模型构建 ==================
def multi_objective_model(x):
    """多目标优化模型（最小化负收益、碳排放、负满意度）"""
    x_ship, x_land = x
    
    # ------ 经济子系统 ------
    # 税收计算
    tax_revenue = (x_ship * base_params['tax_ship'] * 365 * 1.2 * 0.78 +  # 邮轮税（旺季+20%）
                   x_land * base_params['tax_land'] * 365)
    # 隐性成本
    eco_cost = 0.40 * tax_revenue  # 生态修复支出
    housing_subsidy = 0.10 * tax_revenue  # 住房补贴
    
    # 净收益
    net_revenue = tax_revenue - eco_cost - housing_subsidy
    
    # ------ 环境子系统 ------
    # 碳排放约束
    emission = (0.5 * x_ship * 365 + 0.1 * x_land * 365) * (1 - base_params['eta_emission'])
    glacier_retreat = 6.2 - 0.1 * base_params['eco_repair_ratio']  # 冰川退缩速率
    
    # ------ 社会子系统 ------
    # 居民满意度计算
    score = (0.7 * (1 - base_params['housing_cost_ratio']/0.3) - 
             0.1 * (x_ship / 12000) - 
             0.2 * (x_land / 1200))
    
    # ------ 目标函数（加权求和）------
    # 目标1：最大化净收益（转换为最小化负收益）
    # 目标2：最小化碳排放
    # 目标3：最大化满意度（转换为最小化负满意度）
    weights = [0.5, 0.3, 0.2]  # 权重分配
    obj_value = (weights[0] * (-net_revenue/1e6) + 
                 weights[1] * (emission/1e3) + 
                 weights[2] * (-score))
    
    return obj_value

# ================== 约束条件 ==================
constraints = [
    # 财政收入 ≥ 3840万美元
    {'type': 'ineq', 'fun': lambda x: (x[0]*78*365 + x[1]*18*365)/1e6 - 3840},
    
    # 碳排放 ≤ 28000吨
    {'type': 'ineq', 'fun': lambda x: 28000 - (0.5*x[0]*365 + 0.1*x[1]*365)*(1-0.10)},
    
    # 冰川退缩 ≤ 5米/年
    {'type': 'ineq', 'fun': lambda x: 5 - (6.2 - 0.1*0.40)},
    
    # 满意度 ≥ 4.0
    {'type': 'ineq', 'fun': lambda x: (0.7*(1-0.25/0.3) - 0.1*(x[0]/12000) - 0.2*(x[1]/1200)) - 4.0},
    
    # 游客量上限约束
    {'type': 'ineq', 'fun': lambda x: 12000 - x[0]},  # 邮轮容量
    {'type': 'ineq', 'fun': lambda x: 1200 - x[1]}    # 独立游客容量
]

# ================== 模型求解 ==================
# 初始猜测值
x0 = [3000, 1200]

# 变量边界
bounds = [(800, 12000), (500, 1200)]

# 执行优化
result = minimize(
    multi_objective_model,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

# ================== 结果输出 ==================
print("优化结果：")
print(f"邮轮日均量: {result.x[0]:.0f} 人/天")
print(f"独立游客日均量: {result.x[1]:.0f} 人/天")
print(f"目标函数值: {result.fun:.2f}")

# 计算关键指标
tax_rev = (result.x[0]*78*365*1.2*0.78 + result.x[1]*18*365)/1e6
emission_total = (0.5*result.x[0]*365 + 0.1*result.x[1]*365)*(1-0.10)
score = 0.7*(1-0.25/0.3) - 0.1*(result.x[0]/12000) - 0.2*(result.x[1]/1200)

print("\n关键指标：")
print(f"财政收入: {tax_rev:.1f} 百万美元 (阈值≥38.4)")
print(f"碳排放量: {emission_total/1e3:.1f} 千吨 (阈值≤28)")
print(f"居民满意度: {score:.1f}/5.0 (阈值≥4.0)")
print(f"冰川退缩速率: {6.2 - 0.1*0.40:.1f} 米/年 (阈值≤5.0)")

# ================== 灵敏度分析 ==================
def sensitivity_analysis(param, delta=0.2):
    """参数灵敏度分析"""
    original = base_params[param]
    base_params[param] = original * (1 + delta)
    res_pos = minimize(multi_objective_model, x0, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    
    base_params[param] = original * (1 - delta)
    res_neg = minimize(multi_objective_model, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    
    base_params[param] = original  # 恢复原值
    return res_pos.x, res_neg.x

# 示例：分析邮轮税基价影响
print("\n灵敏度分析（邮轮税±20%）：")
tax_ship_pos, tax_ship_neg = sensitivity_analysis('tax_ship')
print(f"税率+20%：邮轮量={tax_ship_pos[0]:.0f}, 独立游客={tax_ship_pos[1]:.0f}")
print(f"税率-20%：邮轮量={tax_ship_neg[0]:.0f}, 独立游客={tax_ship_neg[1]:.0f}")