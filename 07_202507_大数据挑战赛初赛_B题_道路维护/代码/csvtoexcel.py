import pandas as pd

# 1. 读取 CSV 文件（需确保文件路径正确，此处假设文件在当前目录）
#    encoding='utf-8' 确保中文等特殊字符正常读取，若报错可尝试 'gbk'
df = pd.read_csv("DATA/road_maintenance_priority.csv", encoding="utf-8")

df_top30 = df.head(31)
df_top30.to_excel("road_maintenance_priority.xlsx", index=False, engine="openpyxl")
# 2. 保存为 Excel 文件（index=False 表示不保留 pandas 的行索引）
#df.to_excel("road_maintenance_priority.xlsx", index=False, engine="openpyxl")


'''import pandas as pd

# 从CSV文件读取数据
df = pd.read_csv('your_data.csv')

# 只取前30行
df_top30 = df.head(30)

# 保存为Excel文件
df_top30.to_excel('data_top30.xlsx', index=False, engine='openpyxl')'''