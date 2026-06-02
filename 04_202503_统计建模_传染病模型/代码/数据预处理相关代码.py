import pandas as pdimport numpy as npfrom sklearn.preprocessing import StandardScaler
# 读取疫情报告数据（假设数据为csv格式）
domestic_epi_data = pd.read_csv('domestic_epi_data.csv')
international_epi_data = pd.read_csv('international_epi_data.csv')
# 读取人口流动数据
domestic_mobility_data = pd.read_csv('domestic_mobility_data.csv')
international_mobility_data = pd.read_csv('international_mobility_data.csv')
# 读取环境与气候数据
domestic_climate_data = pd.read_csv('domestic_climate_data.csv')
international_climate_data = pd.read_csv('international_climate_data.csv')
# 空间数据网格化# 假设人口数据为 population_data.csv，包含区县级人口信息
population_data = pd.read_csv('population_data.csv')
# 将区县级疫情数据分配至1km×1km的Grid Cellsdef grid_allocation(epi_data, population_data):
    grid_data = []
    for index, row in epi_data.iterrows():
        # 假设区县级疫情数据有'区县代码'字段，人口数据有'区县代码'和'人口数'字段
        county_population = population_data[population_data['区县代码'] == row['区县代码']]['人口数'].values[0]
        grid_population = row['发病数'] * (epi_data['单元格人口数'] / county_population)
        grid_data.append(grid_population)
    return grid_data

domestic_epi_data['网格发病数'] = grid_allocation(domestic_epi_data, population_data)
# 时间序列对齐# 假设疫情数据、人口流动、气候因子数据都有'date'字段表示时间
all_data = pd.merge(domestic_epi_data, domestic_mobility_data, on='date', how='outer')
all_data = pd.merge(all_data, domestic_climate_data, on='date', how='outer')
# 将数据按周进行聚合
all_data['week'] = pd.to_datetime(all_data['date']).dt.isocalendar().week
weekly_data = all_data.groupby('week').agg({
    '发病数':'sum',
    '迁入量':'sum',
    '平均温度':'mean',
    # 其他字段类似处理})
# 缺失值处理# 县级发病数缺失值处理（2015年前部分区县未电子化）def fill_missing_epi(epi_data):
    missing_index = epi_data[epi_data['发病数'].isnull()].index
    for index in missing_index:
        nearby_counties = epi_data[(epi_data['区县代码'] == epi_data.loc[index, '区县代码'] - 1) |
                                  (epi_data['区县代码'] == epi_data.loc[index, '区县代码'] + 1) |
                                  (epi_data['区县代码'] == epi_data.loc[index, '区县代码'] - 2) |
                                  (epi_data['区县代码'] == epi_data.loc[index, '区县代码'] + 2)]
        mean_value = nearby_counties['发病数'].mean()
        epi_data.loc[index, '发病数'] = mean_value
    return epi_data

domestic_epi_data = fill_missing_epi(domestic_epi_data)
# 国际航班数据缺失值处理（疫情期间部分航班未报告）# 假设基于航空公司运营数据补全，这里简单用0填充示例
international_mobility_data = international_mobility_data.fillna(0)
# 医疗资源数据缺失值处理（乡镇卫生院床位数缺失）from sklearn.ensemble import RandomForestRegressor
def fill_missing_medical(medical_data):
    X = medical_data[['人口数', 'GDP', '行政区划等级']]
    y = medical_data['床位数']
    missing_index = medical_data[medical_data['床位数'].isnull()].index

    rf = RandomForestRegressor()
    rf.fit(X.drop(missing_index), y.drop(missing_index))

    medical_data.loc[missing_index, '床位数'] = rf.predict(X.loc[missing_index])
    return medical_data

medical_data = fill_missing_medical(medical_data)
# 数据标准化
scaler = StandardScaler()
standardized_data = scaler.fit_transform(weekly_data)