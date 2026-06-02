import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

# 读取Excel文件
area_df = pd.read_excel('C:/Users/w/Desktop/1.地块编号与面积无类型.xlsx', sheet_name='Sheet1')
crop_df = pd.read_excel('C:/Users/w/Desktop/1.作物数据.xlsx')
data_2023_df = pd.read_excel('C:/Users/w/Desktop/1.2023数据.xlsx')

# 提取信息
land_areas = dict(zip(area_df['地块编号'], area_df['总面积(亩)']))
crops = crop_df['作物编号'].unique()
crop_profits = dict(zip(crop_df['作物编号'], crop_df['亩收益']))
is_bean = crop_df.set_index('作物编号')['是否豆类'].map(lambda x: 1 if x == '是' else 0).to_dict()

# 从作物数据中提取新增的预期销售量和亩产量
expected_sales = dict(zip(crop_df['作物编号'], crop_df['预期销售量']))
yield_per_acre = dict(zip(crop_df['作物编号'], crop_df['亩产量']))

# 2023年种植面积
planting_2023 = data_2023_df.groupby(['地块编号', '作物编号'])['种植面积(亩)'].sum().unstack().fillna(0)

# 初始化问题
problem = LpProblem("Maximize_Profit", LpMaximize)

# 定义变量
years = range(2024, 2031)
plant_vars = LpVariable.dicts("Plant", (years, land_areas.keys(), crops), 0)
use_vars = LpVariable.dicts("Use", (years, land_areas.keys(), crops), 0, 1, LpBinary)

# 生成随机变化
np.random.seed(42)  # 为了结果可重复
other_sales_changes = np.random.uniform(-0.05, 0.02, len(years))
yield_changes = np.random.uniform(-0.02, 0.10, len(years))
cost_growth = 1.05  # 种植成本年增长率
vegetable_price_growth = 1.05  # 蔬菜价格年增长率
mushroom_price_decline = 0.97  # 食用菌价格年下降率
morel_price_decline = 0.95  # 羊肚菌价格年下降率

# 计算每年的预期销售量、亩产量、种植成本和销售价格
future_sales = {year: {} for year in years}
future_yields = {year: {} for year in years}
future_costs = {year: {} for year in years}
future_prices = {year: {} for year in years}

for year_idx, year in enumerate(years):
    for crop in crops:
        # 使用新增的预期销售量
        sales = expected_sales[crop] * (1 + other_sales_changes[year_idx])
        future_sales[year][crop] = sales
        
        # 使用新增的亩产量
        yield_change = yield_per_acre[crop] * (1 + yield_changes[year_idx])
        future_yields[year][crop] = yield_change
        
        # 计算种植成本
        future_costs[year][crop] = cost_growth ** (year_idx + 1)
        
        # 计算销售价格
        if crop in range(1, 17):  # 粮食
            price = crop_profits[crop]
        elif crop in range(17, 38):  # 蔬菜
            price = crop_profits[crop] * (vegetable_price_growth ** (year_idx + 1))
        elif crop in range(38, 42):  # 食用菌
            if crop == 41:  # 假设41是羊肚菌
                price = crop_profits[crop] * (morel_price_decline ** (year_idx + 1))
            else:
                price = crop_profits[crop] * (mushroom_price_decline ** (year_idx + 1))
        
        future_prices[year][crop] = price

# 目标函数：最大化收益
problem += lpSum(
    plant_vars[year][land][crop] * future_prices[year][crop] * future_yields[year][crop] - future_costs[year][crop]
    for year in years 
    for land in land_areas 
    for crop in crops
)

# 约束条件
for year in years:
    for land in land_areas:
        # 面积限制
        problem += lpSum(plant_vars[year][land][crop] for crop in crops) <= land_areas[land]
        
        # E-F地块最多种植两种作物
        if land.startswith('E') or land.startswith('F'):
            problem += lpSum(use_vars[year][land][crop] for crop in crops) <= 2
        else:
            problem += lpSum(use_vars[year][land][crop] for crop in crops) <= 1

    for crop in crops:
        # 销售量限制
        problem += lpSum(plant_vars[year][land][crop] for land in land_areas) <= future_sales[year][crop]

# 连续种植限制
for year in years:
    if year > 2024:
        for land in land_areas:
            for crop in crops:
                problem += plant_vars[year][land][crop] + plant_vars[year-1][land][crop] <= 1

# 三年内种植豆类
for land in land_areas:
    for start_year in range(2024, 2031, 3):
        if start_year + 2 <= 2030:
            problem += lpSum(plant_vars[year][land][crop] * is_bean.get(crop, 0)
                             for year in range(start_year, start_year+3) 
                             for crop in crops) >= 1

# 连接种植变量和使用变量
for year in years:
    for land in land_areas:
        for crop in crops:
            problem += plant_vars[year][land][crop] <= land_areas[land] * use_vars[year][land][crop]

# 新的约束条件
for year in years:
    for land in land_areas:
        # 1-15号作物只能种植在A、B、C地块
        if land not in ['A', 'B', 'C']:
            for crop in range(1, 16):
                problem += plant_vars[year][land][crop] == 0

        # D地块的约束
        if land == 'D':
            problem += lpSum(use_vars[year][land][crop] for crop in range(17, 35)) <= 1
            problem += lpSum(use_vars[year][land][crop] for crop in range(35, 38)) <= 1
            problem += use_vars[year][land][16] + lpSum(use_vars[year][land][crop] for crop in range(17, 35)) <= 1

        # 17-34和38-41只能种植在E、F地块
        if land not in ['E', 'F']:
            for crop in list(range(17, 35)) + list(range(38, 42)):
                problem += plant_vars[year][land][crop] == 0

# 求解问题
problem.solve()

# 生成结果和计算利润
results = []
annual_profit = {}
annual_details = []

for year in years:
    total_profit = 0
    for land in land_areas:
        for crop in crops:
            if plant_vars[year][land][crop].varValue > 0:
                area = plant_vars[year][land][crop].varValue
                profit = area * future_prices[year][crop] * future_yields[year][crop] - future_costs[year][crop]
                total_profit += profit
                results.append({
                    '年份': year,
                    '地块编号': land,
                    '作物编号': crop,
                    '种植面积(亩)': area
                })
                
    annual_profit[year] = total_profit
    print(f"Year {year}: Total Profit = {total_profit}")

    # 保存每年的详细信息
    for crop in crops:
        annual_details.append({
            '年份': year,
            '作物编号': crop,
            '预期销售量': future_sales[year][crop],
            '亩产量变化': future_yields[year][crop],
            '种植成本变化': future_costs[year][crop],
            '销售价格变化': future_prices[year][crop]
        })

# 保存结果到Excel
result_df = pd.DataFrame(results)
result_df.to_excel('C:/Users/w/Desktop/未来七年种植方案.xlsx', index=False)

# 保存详细信息到Excel
details_df = pd.DataFrame(annual_details)
details_df.to_excel('C:/Users/w/Desktop/未来七年农作物详细信息.xlsx', index=False)
