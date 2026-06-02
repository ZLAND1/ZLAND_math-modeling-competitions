import pandas as pd

# 读取数据
file_path = 'C:/Users/lenovo/Desktop/2023各量(2).xlsx'
data = pd.read_excel(file_path)

# 确保数据中没有缺失值
data = data.dropna()

# 创建包含需要计算相关性的特征矩阵
features = ['预期销量', '销售价格', '种植单位成本']
feature_matrix = data[features]

# 计算相关性矩阵
correlation_matrix = feature_matrix.corr()

# 输出相关性矩阵
print("预期销量、销售价格与种植单位成本之间的相关性矩阵:")
print(correlation_matrix)

# 保存相关性矩阵到Excel文件
output_file_path = 'C:/Users/lenovo/Desktop/feature_correlation_matrix.xlsx'
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    correlation_matrix.to_excel(writer, sheet_name='特征相关性矩阵')

print(f"特征相关性矩阵已保存到 {output_file_path}")
