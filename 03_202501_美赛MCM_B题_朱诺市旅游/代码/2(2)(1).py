import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ================== 数据模拟 ==================
np.random.seed(42)

# 时间范围
days = 365
months = 12

# 基础参数（更新系数）
base_params = {
    'x_ship': 3000,
    'x_land': 1200,
    'eta_emission': 0.10,
    'eco_repair_ratio': 0.40,    # 生态修复支出占比
    'tax_ship': 78,             
    'tax_land': 18,
    'ship_co2': 0.01913,         # 邮轮碳排放系数（吨/人）
    'land_co2': 0.01120          # 陆地游客碳排放系数（吨/人）
}

# ================== 模型构建 ==================
def multi_objective_model(x):
    """多目标优化模型（经济优先，环境社会约束）"""
    x_ship, x_land = x
    
    # ------ 经济子系统（含旺季税率）------
    # 旺季天数计算（6-8月共92天）
    tax_ship_season = x_ship * base_params['tax_ship'] * 1.2 * 92
    tax_ship_off = x_ship * base_params['tax_ship'] * (365-92)
    tax_land_total = x_land * base_params['tax_land'] * 365
    tax_revenue = tax_ship_season + tax_ship_off + tax_land_total
    
    # 支出计算
    eco_cost = tax_revenue * base_params['eco_repair_ratio']
    housing_subsidy = tax_revenue * 0.10
    net_revenue = tax_revenue - eco_cost - housing_subsidy
    
    # ------ 环境子系统 ------
    emission = (x_ship*365*base_params['ship_co2'] + x_land*365*base_params['land_co2']) * (1 - base_params['eta_emission'])
    
    # ------ 社会子系统 ------
    satisfaction = 3.8 + 0.35*(base_params['eco_repair_ratio']/0.4) + 0.25 - 0.1*(x_ship/3000 + x_land/1200)
    
    # 目标函数：最小化碳排放（经济目标通过约束保证）
    return emission / 1e3  # 转换为千吨

# ================== 约束条件 ==================
constraints = [
    # 财政收入 ≥ 3840万美元
    {'type': 'ineq', 'fun': lambda x: ((x[0]*78*1.2*92 + x[0]*78*(365-92) + x[1]*18*365)/1e6) - 38.4},
    
    # 碳排放 ≤ 28000吨
    {'type': 'ineq', 'fun': lambda x: 28.0 - ((x[0]*365*0.01913 + x[1]*365*0.0112)*(1-0.10))/1e3},
    
    # 冰川退缩 ≤5米/年
    {'type': 'ineq', 'fun': lambda _: 5.0 - (6.2 - base_params['eco_repair_ratio']*100*0.035)},
    
    # 满意度 ≥4.0
    {'type': 'ineq', 'fun': lambda x: (3.8 + 0.35 + 0.25 - 0.1*(x[0]/3000 + x[1]/1200)) - 4.0},
    
    # 运力约束
    {'type': 'ineq', 'fun': lambda x: 12000 - x[0]},  # 邮轮最大容量
    {'type': 'ineq', 'fun': lambda x: 1200 - x[1]}    # 独立游客最大容量
]

# ================== 模型求解 ==================
x0 = [3000, 1200]
bounds = [(2500, 12000), (800, 1200)]

result = minimize(
    multi_objective_model,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-6}
)

# ================== 结果输出 ==================
print("\n优化结果：")
print(f"邮轮日均量: {result.x[0]:.0f} 人/天")
print(f"独立游客日均量: {result.x[1]:.0f} 人/天")

# 详细指标计算
tax_ship = (result.x[0]*78*1.2*92 + result.x[0]*78*(365-92))/1e6
tax_land = result.x[1]*18*365/1e6
emission = (result.x[0]*365*0.01913 + result.x[1]*365*0.0112)*0.9
glacier_retreat = 6.2 - base_params['eco_repair_ratio']*100*0.035
satisfaction = 3.8 + 0.35 + 0.25 - 0.1*(result.x[0]/3000 + result.x[1]/1200)

print("\n关键指标验证：")
print(f"1. 财政收入: {tax_ship + tax_land:.2f} 百万美元 (目标≥38.4)")
print(f"2. 碳排放量: {emission/1e3:.2f} 千吨 (目标≤28.0)")
print(f"3. 冰川退缩: {glacier_retreat:.1f} 米/年 (目标≤5.0)")
print(f"4. 居民满意度: {satisfaction:.2f}/5.0 (目标≥4.0)")
print(f"5. 游客承载量: 邮轮{result.x[0]/12000:.1%}，独立{result.x[1]/1200:.1%}")

# ================== 敏感性分析 ==================
print("\n策略验证：")
print("旺季邮轮税上浮20%：实际税率 ${:.1f}/人".format(78*1.2))
print("Mendenhall冰川限流：{}人/天".format(int(result.x[0]*0.2)))  # 假设20%游客访问冰川