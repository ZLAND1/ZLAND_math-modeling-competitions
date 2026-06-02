# -*- coding: utf-8 -*-

#本代码仅进行了查验两次数据是否一致
import time
import sys
import re
import io
import pandas as pd
import numpy as np
import hashlib
import os
from tqdm import tqdm  # 进度条，方便查看大文件处理进度
import matplotlib.pyplot as plt

# 解决终端中文显示乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# -------------------------- 1. 配置参数（无需修改，按您的路径设置） --------------------------
file1_path = r"D:\数学建模比赛\大数据挑战赛复赛2025826_829\DATA\train_data.csv"
file2_path = r"D:\数学建模比赛\大数据挑战赛复赛2025826_829\DATA\data.csv"
expected_rows = 840000  # 预期行数（含表头，若表头不存在需改为840000）
expected_cols = 10       # 预期列数
output_dir = r"D:\数学建模比赛\大数据挑战赛复赛2025826_829\DATA"  # 差异报告输出路径

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------- 2. 工具函数（辅助比对） --------------------------
def calculate_file_hash(file_path, chunk_size=1024*1024):
    """计算文件的MD5哈希值，快速判断两文件是否完全一致"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def compare_numeric_values(val1, val2, tol=1e-6):
    """比较数值型数据（支持浮点数微小误差，如1.234和1.2340001视为一致）"""
    # 先判断是否均为数值（排除字符串等非数值类型）
    if not (np.issubdtype(type(val1), np.number) and np.issubdtype(type(val2), np.number)):
        return val1 == val2
    # 浮点数按容忍度比对，整数直接相等比对
    if isinstance(val1, float) or isinstance(val2, float):
        return abs(val1 - val2) <= tol
    else:
        return val1 == val2

# -------------------------- 3. 第一步：基础校验（列名、维度） --------------------------
print("="*50)
print("开始基础校验：列名一致性 + 数据维度校验")
print("="*50)

# 读取两文件（大文件读取优化：设置low_memory=False避免类型推断错误）
try:
    df1 = pd.read_csv(file1_path, low_memory=False)
    df2 = pd.read_csv(file2_path, low_memory=False)
except FileNotFoundError as e:
    print(f"错误：文件未找到！{e}")
    exit()
except Exception as e:
    print(f"错误：读取文件失败！{e}")
    exit()

# 3.1 列名一致性校验
col1 = set(df1.columns.tolist())
col2 = set(df2.columns.tolist())
if col1 != col2:
    missing_in_file1 = col2 - col1
    missing_in_file2 = col1 - col2
    print(f"❌ 列名不一致！")
    print(f"   data.csv 有但 train_data.csv 没有的列：{missing_in_file1}")
    print(f"   train_data.csv 有但 data.csv 没有的列：{missing_in_file2}")
else:
    print(f"✅ 列名完全一致！列名列表：{df1.columns.tolist()}")

# 3.2 数据维度校验（行数、列数）
print(f"\n【train_data.csv】维度：{df1.shape[0]} 行 × {df1.shape[1]} 列")
print(f"【data.csv】维度：{df2.shape[0]} 行 × {df2.shape[1]} 列")
print(f"【预期维度】：{expected_rows} 行 × {expected_cols} 列")

# 检查列数是否符合预期
if df1.shape[1] != expected_cols or df2.shape[1] != expected_cols:
    print(f"❌ 列数不符合预期！train_data.csv 列数：{df1.shape[1]}，data.csv 列数：{df2.shape[1]}")
else:
    print(f"✅ 列数均符合预期（10列）")

# 检查行数是否符合预期
if df1.shape[0] != expected_rows or df2.shape[0] != expected_rows:
    print(f"❌ 行数不符合预期！train_data.csv 行数：{df1.shape[0]}，data.csv 行数：{df2.shape[0]}")
else:
    print(f"✅ 行数均符合预期（840001行）")

# 若列名不一致，直接终止后续比对（需先修正列名）
if col1 != col2:
    print("\n⚠️  列名不一致，无法进行后续内容比对，请先统一列名！")
    exit()

# -------------------------- 4. 第二步：快速全量比对（哈希值） --------------------------
print("\n" + "="*50)
print("开始快速全量比对：文件MD5哈希值校验")
print("="*50)

hash1 = calculate_file_hash(file1_path)
hash2 = calculate_file_hash(file2_path)

if hash1 == hash2:
    print(f"✅ 两文件MD5哈希值完全一致！")
    print(f"   train_data.csv 哈希值：{hash1}")
    print(f"   data.csv 哈希值：{hash2}")
    print("\n结论：两文件内容完全一致，无需后续细节比对！")
    exit()
else:
    print(f"❌ 两文件MD5哈希值不一致！")
    print(f"   train_data.csv 哈希值：{hash1}")
    print(f"   data.csv 哈希值：{hash2}")
    print("⚠️  需进行细节比对，定位具体差异位置...")

# -------------------------- 5. 第三步：细节比对（定位差异行/列） --------------------------
print("\n" + "="*50)
print("开始细节比对：定位具体差异（共840000行，耗时可能较长，请耐心等待）")
print("="*50)

# 5.1 统一列顺序（确保按相同列顺序比对）
df2 = df2[df1.columns]

# 5.2 初始化差异记录
diff_records = []  # 存储差异详情：[行号, 列名, train_data值, data值]
diff_rows = set()  # 存储有差异的行号（去重）

# 5.3 逐列比对（按列循环，减少内存占用）
for col in tqdm(df1.columns, desc="按列比对进度"):
    # 提取当前列数据
    col1_data = df1[col].values
    col2_data = df2[col].values
    
    # 比对每一行（跳过NaN与NaN的比对，视为一致）
    for row_idx in range(len(col1_data)):
        val1 = col1_data[row_idx]
        val2 = col2_data[row_idx]
        
        # 处理NaN情况：两者均为NaN视为一致，否则视为差异
        if pd.isna(val1) and pd.isna(val2):
            continue
        elif pd.isna(val1) or pd.isna(val2):
            diff_records.append([row_idx, col, val1, val2])
            diff_rows.add(row_idx)
            continue
        
        # 比对数值/字符串（支持浮点数容忍度）
        if not compare_numeric_values(val1, val2):
            diff_records.append([row_idx, col, val1, val2])
            diff_rows.add(row_idx)

# 5.4 输出差异统计
print(f"\n【差异汇总】")
print(f"总差异条数：{len(diff_records)} 条")
print(f"有差异的行数：{len(diff_rows)} 行")

# 若有差异，输出前10条示例并保存全量差异报告
if len(diff_records) > 0:
    # 输出前10条差异示例
    print(f"\n前10条差异示例（行号从0开始，对应数据行，0为表头行）：")
    diff_df_sample = pd.DataFrame(diff_records[:10], 
                                 columns=["行号", "列名", "train_data.csv 值", "data.csv 值"])
    print(diff_df_sample.to_string(index=False))
    
    # 保存全量差异报告到CSV
    diff_df_full = pd.DataFrame(diff_records, 
                               columns=["行号", "列名", "train_data.csv 值", "data.csv 值"])
    diff_report_path = os.path.join(output_dir, "数据差异报告.csv")
    diff_df_full.to_csv(diff_report_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 全量差异报告已保存至：{diff_report_path}")

    # 统计各列差异次数（高频差异列）
    col_diff_count = diff_df_full["列名"].value_counts()
    print(f"\n【各列差异次数统计】")
    print(col_diff_count)

    # 可视化高频差异列（前5列）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.figure(figsize=(12, 6))
    col_diff_count.head(5).plot(kind="bar", color="#ff6b6b")
    plt.title("各列差异次数统计（前5列）")
    plt.xlabel("列名")
    plt.ylabel("差异次数")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "列差异统计图表.png"), dpi=300)
    print(f"✅ 列差异统计图表已保存至：{os.path.join(output_dir, '列差异统计图表.png')}")

# -------------------------- 6. 第四步：特殊值校验（缺失值、异常值） --------------------------
print("\n" + "="*50)
print("开始特殊值校验：缺失值 + 异常值一致性")
print("="*50)

# 6.1 缺失值比对（各列缺失值数量）
missing1 = df1.isnull().sum()
missing2 = df2.isnull().sum()
missing_diff = missing1 - missing2

print("【各列缺失值数量】")
missing_compare = pd.DataFrame({
    "train_data.csv 缺失数": missing1,
    "data.csv 缺失数": missing2,
    "缺失数差异（train - data）": missing_diff
})
print(missing_compare.to_string())

if (missing_diff != 0).any():
    print(f"❌ 缺失值数量不一致！差异列：{missing_diff[missing_diff != 0].index.tolist()}")
else:
    print(f"✅ 所有列缺失值数量完全一致！")

# 6.2 异常值比对（以数值列为例，用IQR法则检测异常值）
print(f"\n【数值列异常值比对（IQR法则：Q1-1.5IQR ~ Q3+1.5IQR 外视为异常）】")
numeric_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
outlier_diff = {}

for col in numeric_cols:
    # 计算train_data.csv的异常值
    q1_1 = df1[col].quantile(0.25)
    q3_1 = df1[col].quantile(0.75)
    iqr_1 = q3_1 - q1_1
    outlier1_count = ((df1[col] < q1_1 - 1.5*iqr_1) | (df1[col] > q3_1 + 1.5*iqr_1)).sum()
    
    # 计算data.csv的异常值
    q1_2 = df2[col].quantile(0.25)
    q3_2 = df2[col].quantile(0.75)
    iqr_2 = q3_2 - q1_2
    outlier2_count = ((df2[col] < q1_2 - 1.5*iqr_2) | (df2[col] > q3_2 + 1.5*iqr_2)).sum()
    
    outlier_diff[col] = outlier1_count - outlier2_count
    print(f"列 {col}：train_data 异常值 {outlier1_count} 个，data.csv 异常值 {outlier2_count} 个，差异 {outlier_diff[col]} 个")

if any(v != 0 for v in outlier_diff.values()):
    print(f"❌ 数值列异常值数量不一致！差异列：{[k for k, v in outlier_diff.items() if v != 0]}")
else:
    print(f"✅ 所有数值列异常值数量完全一致！")

# -------------------------- 7. 最终结论 --------------------------
print("\n" + "="*50)
print("最终比对结论")
print("="*50)
if hash1 == hash2:
    print("✅ 两文件完全一致（MD5哈希值相同，无任何差异）")
else:
    print(f"❌ 两文件存在差异，具体如下：")
    print(f"   1. 总差异条数：{len(diff_records)} 条")
    print(f"   2. 有差异的行数：{len(diff_rows)} 行")
    print(f"   3. 缺失值/异常值：{'存在不一致' if (missing_diff != 0).any() or any(v != 0 for v in outlier_diff.values()) else '完全一致'}")
    print(f"   4. 详细差异请查看：{os.path.join(output_dir, '数据差异报告.csv')}")