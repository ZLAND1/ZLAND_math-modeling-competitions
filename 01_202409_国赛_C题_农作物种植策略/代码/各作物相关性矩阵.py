import pandas as pd
import numpy as np

# 读取数据
file_path = 'C:/Users/lenovo/Desktop/产量售价成本.xlsx'
data = pd.read_excel(file_path)

# 确保数据中没有缺失值
data = data.dropna()

# 创建特征矩阵
feature_matrix = data.pivot_table(index='作物名称', values=['亩产量', '销售价格', '种植单位成本'])

# 计算作物之间的相关性矩阵
# 转置特征矩阵以便计算不同作物之间的相关性
crop_correlation_matrix = feature_matrix.T.corr()

# 输出作物之间的相关性矩阵
print("作物之间的相关性矩阵:")
print(crop_correlation_matrix)

# 保存相关性矩阵到Excel文件
output_file_path = 'C:/Users/lenovo/Desktop/crop_correlation_matrix.xlsx'
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    crop_correlation_matrix.to_excel(writer, sheet_name='作物相关性矩阵')

print(f"作物之间的相关性矩阵已保存到 {output_file_path}")
