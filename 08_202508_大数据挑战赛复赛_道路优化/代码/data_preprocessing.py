# -*- coding: utf-8 -*-

#本代码仅进行了缺失值检查、分类特征的量化、以及列号的重命名

import pandas as pd
import numpy as np
import time
import sys
import re
import io

# 解决终端中文显示乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def preprocess_road_data(csv_path, output_excel_path=None, output_csv_path=None):
    """
    仅进行必要格式转换，不处理任何异常值，保留原始数据特征
    核心操作：读取数据→格式转换→保留原始值→统计缺失值→输出
    """
    start_time = time.time()
    print("开始数据处理（仅复制和必要转换，不处理异常值）...")
    
    # 1. 读取原始数据（完整保留所有值，包括异常值和缺失值）
    chunk_size = 100000
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, keep_default_na=True):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"数据读取完成，共{df.shape[0]}行，{df.shape[1]}列")
    
    # 2. 重命名列名（简写，便于后续处理）
    column_mapping = {
        'Segment ID': 'ID',
        'PCI': 'PCI',
        'Road Type': 'RT',
        'AADT': 'AADT',
        'Asphalt Type': 'AT',
        'Last Maintenance': 'LM',
        'Average Rainfall': 'AR',
        'Rutting': 'R',
        'IRI': 'IRI',
        'Needs Maintenance': 'NM'
    }
    df = df.rename(columns=column_mapping)
    
    # 3. 处理ID列：去除开头的"SID"前缀（仅格式转换，不修改数字）
    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str).apply(
            lambda x: re.sub(r'^SID', '', x, flags=re.IGNORECASE)
        )
        try:
            df['ID'] = pd.to_numeric(df['ID'], downcast='integer')
            print("ID列已去除'SID'前缀并转换为整数类型")
        except ValueError:
            print("ID列已去除'SID'前缀，部分ID含非数字字符，保留为字符串类型")
    else:
        print("警告：数据中未找到'ID'列，跳过ID处理")
    
    # 4. 分类特征量化（仅映射文本到数字，不处理异常值）
    # 4.1 Road Type量化：Primary(3) > Secondary(2) > Tertiary(1)
    rt_mapping = {'Primary': 3, 'Secondary': 2, 'Tertiary': 1}
    df['RT'] = df['RT'].map(rt_mapping)  # 原始缺失值保持为NaN
    
    # 4.2 Asphalt Type量化：Concrete(1) < Asphalt(2)
    at_mapping = {'Concrete': 1, 'Asphalt': 2}
    df['AT'] = df['AT'].map(at_mapping)  # 原始缺失值保持为NaN
    
    # 5. 时间特征转换：计算维护间隔年数（仅数学转换，不处理异常值）
    df['YSM'] = 2025 - df['LM']  # 若LM为NaN，YSM保持为NaN
    df = df.drop('LM', axis=1)  # 删除原始年份列
    
    # 6. 缺失值统计（仅统计，不填充）
    print("\n===== 缺失值统计结果 =====")
    numeric_features = ['PCI', 'AADT', 'AR', 'R', 'IRI', 'YSM']
    categorical_features = ['RT', 'AT']
    all_features = numeric_features + categorical_features + ['ID', 'NM']
    
    missing_counts = df[all_features].isnull().sum()  # 各特征缺失值数量
    missing_ratio = (missing_counts / len(df)).round(4) * 100  # 缺失比例（%）
    
    # 整理为表格并排序
    missing_stats = pd.DataFrame({
        '特征': missing_counts.index,
        '缺失值数量': missing_counts.values,
        '缺失比例(%)': missing_ratio.values
    }).sort_values(by='缺失值数量', ascending=False)
    print(missing_stats)
    
    # 总缺失值统计
    total_missing = missing_counts.sum()
    total_cells = len(df) * len(all_features)
    print(f"\n总缺失值数量：{total_missing}")
    print(f"总缺失比例：{round((total_missing / total_cells) * 100, 2)}%")
    
    # 7. 保存数据（完整复制原始值，仅转换格式）
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"\n数据已保存至CSV文件: {output_csv_path}（未处理异常值，保留原始数据）")
    
    if output_excel_path:
        if len(df) > 10000:
            df_sample = df.sample(10000, random_state=42)
            df_sample.to_excel(output_excel_path, index=False)
            print(f"采样10000行数据已保存至Excel文件: {output_excel_path}（未处理异常值）")
        else:
            df.to_excel(output_excel_path, index=False)
            print(f"数据已保存至Excel文件: {output_excel_path}（未处理异常值）")
    
    # 输出处理时间
    end_time = time.time()
    print(f"\n数据处理完成，总耗时: {end_time - start_time:.2f}秒")
    
    return df

# 使用示例
if __name__ == "__main__":
    input_csv = "DATA/data.csv"
    output_excel = "DATA/processed_road_data_sample.xlsx"
    output_csv = "DATA/processed_road_data.csv"
    
    processed_df = preprocess_road_data(
        csv_path=input_csv,
        output_excel_path=output_excel,
        output_csv_path=output_csv
    )
    
    if processed_df is not None:
        print("\n处理后的数据前5行（保留原始值，未处理异常值）:")
        print(processed_df.head())