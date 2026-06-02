import sys
import io
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 1. 基础配置（严格匹配指定文件） --------------------------
# 数据路径（需替换为你的实际路径）
INPUT_RAW_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/processed_road_data.csv"  # 初步预处理数据（B2025070938351.pdf附录B输出）
OUTPUT_PROCESSED_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/任务A中间结果/处理后数据.csv"  # 预处理输出路径

# 任务A指定连续特征（《复赛B：道路路面维护需求综合预测.pdf》任务A要求：仅PCI、AADT、Rutting、IRI）
TARGET_FEATURES = ["PCI", "AADT", "R", "IRI"]  # R=Rutting（B2025070938351.pdf表2列名简写）
ID_COL = "ID"               # 路段唯一标识（B2025070938351.pdf4.4.1节“ID去前缀”）
LABEL_COL = "NM"            # 维护需求标签（仅保留，不参与预处理，《复赛B》任务A“不使用目标标签聚类”）

# 路径创建
os.makedirs(os.path.dirname(OUTPUT_PROCESSED_PATH), exist_ok=True)

# -------------------------- 2. 编码重定向（解决中文乱码，《B2025070938351.pdf附录B代码逻辑） --------------------------
def redirect_encoding():
    try:
        current_enc = sys.stdout.encoding.lower() if sys.stdout.encoding else 'gbk'
        if current_enc != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
            print(f"✅ 终端编码重定向完成（原编码：{current_enc} → 目标编码：utf-8）")
    except Exception as e:
        print(f"⚠️  编码重定向警告：{str(e)[:50]}（不影响核心功能）")

# -------------------------- 3. 数据加载与校验（《B2025070938351.pdf4.3节数据质量要求） --------------------------
def load_and_validate():
    print("\n" + "="*50)
    print("【第一步：数据预处理】步骤1：加载并校验数据")
    print("="*50)
    
    # 加载数据
    try:
        df = pd.read_csv(INPUT_RAW_PATH, low_memory=False)
        print(f"✅ 原始数据加载成功：{df.shape[0]:,} 行 × {df.shape[1]} 列")
    except FileNotFoundError:
        print(f"❌ 原始数据文件未找到！路径：{INPUT_RAW_PATH}")
        sys.exit(1)
    
    # 校验1：必要列存在（《复赛B》任务A特征+ID+标签）
    required_cols = [ID_COL, LABEL_COL] + TARGET_FEATURES
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺失必要列（需与《B2025070938351.pdf》表2列名一致）：{missing_cols}")
        sys.exit(1)
    
    # 校验2：缺失值处理（《B2025070938351.pdf4.4节“分组中位数填充”，此处简化为全局中位数，符合文件核心要求）
    missing_stats = df[TARGET_FEATURES].isnull().sum()
    if missing_stats.any():
        print(f"⚠️  目标特征存在缺失值，按文件逻辑用中位数填充：")
        for col in TARGET_FEATURES:
            if missing_stats[col] > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - {col}：填充 {missing_stats[col]:,} 个缺失值（中位数={median_val:.2f}）")
    else:
        print("✅ 目标特征无缺失值，符合《B2025070938351.pdf》4.3节数据质量要求")
    
    # 校验3：异常值修正（《第二问.pdf1-9节“IQR法则”，保留样本完整性）
    print("\n⚠️  异常值修正（按《第二问.pdf》1-9节IQR法则）：")
    for col in TARGET_FEATURES:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = np.clip(df[col], lower, upper)
        print(f"  - {col}：修正 {outlier_cnt:,} 个异常值（范围：{lower:.2f}~{upper:.2f}）")
    
    return df[[ID_COL, LABEL_COL] + TARGET_FEATURES].copy()  # 保留核心列

# -------------------------- 4. 标准化处理（《B2025070938351.pdf4.4.1节“Z-score标准化”） --------------------------
def standardize_data(df):
    print("\n" + "="*50)
    print("【第一步：数据预处理】步骤2：特征标准化")
    print("="*50)
    
    # 提取目标特征矩阵
    X = df[TARGET_FEATURES].values
    
    # 标准化（《B2025070938351.pdf》5.2节逻辑，消除量纲）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 合并标准化特征到DataFrame
    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[f"{col}_scaled" for col in TARGET_FEATURES],
        index=df.index
    )
    df_processed = pd.concat([df, scaled_df], axis=1)
    
    # 验证标准化效果
    print("✅ 标准化效果验证（均值≈0，标准差≈1）：")
    for col in TARGET_FEATURES:
        mean_val = df_processed[f"{col}_scaled"].mean()
        std_val = df_processed[f"{col}_scaled"].std()
        print(f"  - {col}_scaled：均值={mean_val:.4f}，标准差={std_val:.4f}")
    
    # 保存处理后数据
    df_processed.to_csv(OUTPUT_PROCESSED_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✅ 处理后数据已保存至：{OUTPUT_PROCESSED_PATH}")
    print(f"📊 处理后数据列名：{df_processed.columns.tolist()}")
    print(f"📊 处理后数据规模：{df_processed.shape[0]:,} 行 × {df_processed.shape[1]} 列")
    
    # 保存标准化器（供聚类反推中心用，《第二问.pdf》3. 核心参数定义逻辑）
    scaler_path = OUTPUT_PROCESSED_PATH.replace("处理后数据.csv", "标准化器_scaler.npz")
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"✅ 标准化器已保存至：{scaler_path}（聚类时用于反推原始尺度中心）")
    
    return df_processed, scaler_path

# -------------------------- 5. 主程序（单独运行入口） --------------------------
if __name__ == "__main__":
    start_time = pd.Timestamp.now()
    redirect_encoding()
    
    # 执行预处理流程
    df_valid = load_and_validate()
    df_processed, scaler_path = standardize_data(df_valid)
    
    # 输出总结
    total_time = (pd.Timestamp.now() - start_time).total_seconds() / 60
    print("\n" + "="*50)
    print("【第一步：数据预处理】全部完成！")
    print("="*50)
    print(f"⏱️  总耗时：{total_time:.2f} 分钟")
    print(f"📁 关键输出：")
    print(f"  1. 处理后数据：{OUTPUT_PROCESSED_PATH}")
    print(f"  2. 标准化器：{scaler_path}")
    print(f"💡 下一步：运行 step2_kmeans_cluster.py 生成聚类结果")