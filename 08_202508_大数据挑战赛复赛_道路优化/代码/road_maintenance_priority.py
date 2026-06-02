# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import re
import io
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 1. 全局配置中文字体（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题


# 解决终端中文显示乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ----------------------
# 1. 加载预处理后的数据
# ----------------------
def load_processed_data(csv_path="DATA/processed_road_data.csv"):
    print("加载预处理后的数据...")
    df = pd.read_csv(csv_path)
    # 保留核心分析字段（排除ID和目标变量NM）
    analysis_cols = ['PCI', 'R', 'AR', 'YSM', 'RT', 'AT', 'AADT']
    df = df[analysis_cols].dropna()  # 理论上无缺失，保险处理
    print(f"分析数据形状：{df.shape}（行：样本数，列：指标数）")
    return df

# ----------------------
# 2. 三维度指标划分与标准化
# ----------------------
def split_and_standardize(df):
    # 定义三维度指标（基于业务逻辑和SHAP分析）
    dimensions = {
        "结构型": ['PCI', 'R'],  # 路面结构破损（PCI越低越差，R越高越差）
        "环境型": ['AR', 'YSM'],  # 环境加速老化（AR越高、YSM越长越差）
        "效率型": ['RT', 'AT', 'AADT']  # 使用效率（RT越高、AADT越大越需优先）
    }
    
    # 标准化：每个维度内单独标准化（避免跨维度量纲影响）
    scaled_data = {}
    scalers = {}  # 保存标准化器，用于后续更新数据
    
    for dim, cols in dimensions.items():
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cols])
        scaled_data[dim] = pd.DataFrame(scaled, columns=cols)
        scalers[dim] = scaler  # 保存标准化器
        print(f"维度[{dim}]标准化完成，指标：{cols}")
    
    return scaled_data, dimensions, scalers

# ----------------------
# 3. 分维度主成分分析（PCA）
# ----------------------
def pca_per_dimension(scaled_data, dimensions):
    pca_results = {}
    pca_models = {}  # 保存PCA模型，用于动态更新
    
    for dim, cols in dimensions.items():
        data = scaled_data[dim].values
        # 每个维度提取1个主成分（解释核心变异）
        pca = PCA(n_components=1)
        pc = pca.fit_transform(data)
        # 调整主成分方向：使主成分值越高，风险越大（符合业务逻辑）
        if dim == "结构型":
            # PCI与风险负相关（需反转方向），R与风险正相关
            # 检查载荷：若PCI载荷为正，主成分需乘-1（确保主成分越高风险越大）
            if pca.components_[0][0] > 0:  # PCI的载荷
                pc = -pc
        # 环境型和效率型指标与风险正相关，无需反转
        pca_results[dim] = pd.Series(pc.flatten(), name=f"PC_{dim}")
        pca_models[dim] = pca
        
        # 输出主成分解释力
        explained = pca.explained_variance_ratio_[0] * 100
        print(f"维度[{dim}]主成分解释率：{explained:.2f}%（保留核心信息）")
    
    # 合并主成分结果为DataFrame
    pc_df = pd.concat(pca_results.values(), axis=1)
    return pc_df, pca_models

# ----------------------
# 4. 熵权法计算维度权重
# ----------------------
def entropy_weight(pc_df):
    # 计算每个主成分的熵值（熵越小，离散度越高，权重越大）
    def calc_entropy(series):
        # 标准化到[0,1]避免log(0)
        normalized = (series - series.min()) / (series.max() - series.min() + 1e-8)
        p = normalized / normalized.sum()  # 概率分布
        entropy = -np.sum(p * np.log(p + 1e-8))  # 信息熵
        return entropy
    
    # 计算各主成分的熵值
    entropies = {col: calc_entropy(pc_df[col]) for col in pc_df.columns}
    
    # 计算权重：(1-熵值)/总和
    weights = {col: (1 - ent) for col, ent in entropies.items()}
    total = sum(weights.values())
    weights = {col: w / total for col, w in weights.items()}
    
    print("\n===== 熵权法计算维度权重 =====")
    for col, w in weights.items():
        print(f"{col} 权重：{w:.4f}（熵值：{entropies[col]:.4f}）")
    return weights

# ----------------------
# 5. 优先级划分（基于K-means和业务规则）
# ----------------------
def assign_priority(pc_df, weights):
    # 计算综合风险值S = 结构型权重*PC_结构型 + 环境型权重*PC_环境型 + 效率型权重*PC_效率型
    pc_df['S'] = (
        weights['PC_结构型'] * pc_df['PC_结构型'] +
        weights['PC_环境型'] * pc_df['PC_环境型'] +
        weights['PC_效率型'] * pc_df['PC_效率型']
    )
    
    # 用K-means聚类确定阈值（4类：紧急、高、中、低）
    kmeans = KMeans(n_clusters=4, random_state=42)
    pc_df['cluster'] = kmeans.fit_predict(pc_df[['S']])
    
    # 按聚类中心排序，确定优先级（中心越高，优先级越高）
    cluster_centers = pd.DataFrame({
        'cluster': range(4),
        'center': kmeans.cluster_centers_.flatten()
    }).sort_values('center', ascending=False)
    
    # 映射聚类到优先级
    priority_map = {row['cluster']: i+1 for i, row in cluster_centers.iterrows()}
    pc_df['priority_raw'] = pc_df['cluster'].map(priority_map)  # 1=紧急，2=高，3=中，4=低
    
    # 结合业务规则微调（主干道优先）
    # 注：RT=3（Primary）的道路，若raw=2（高），提升为1（紧急）
    # 需关联原始数据中的RT（道路类型）
    # 此处从原df提取RT（确保索引一致）
    # 假设df已通过参数传入，这里简化处理（实际使用时需补充）
    # pc_df['RT'] = df['RT'].values
    # pc_df.loc[pc_df['RT'] == 3, 'priority_raw'] = pc_df.loc[pc_df['RT'] == 3, 'priority_raw'].apply(lambda x: max(1, x-1))
    
    # 重命名优先级
    priority_names = {1: '紧急', 2: '高', 3: '中', 4: '低'}
    pc_df['priority'] = pc_df['priority_raw'].map(priority_names)
    
    print("\n===== 优先级划分结果 =====")
    print(pc_df['priority'].value_counts(normalize=True).round(4) * 100)
    return pc_df

# ----------------------
# 6. 动态更新机制（模拟季度更新）
# ----------------------
def dynamic_update_demo(original_df, scalers, pca_models, weights):
    # 模拟季度新数据（随机抽取10%数据，添加噪声模拟变化）
    np.random.seed(43)
    sample_size = int(len(original_df) * 0.1)
    new_data = original_df.sample(sample_size, random_state=42).copy()
    
    # 模拟环境型指标变化（雨季AR增加10%）
    new_data['AR'] = new_data['AR'] * 1.1
    # 模拟结构型指标变化（车辙深度R随时间增加5%）
    new_data['R'] = new_data['R'] * 1.05
    
    print("\n===== 动态更新模拟（季度数据变化） =====")
    print(f"模拟更新{sample_size}条数据（AR+10%，R+5%）")
    
    # 标准化新数据（使用历史scaler）
    scaled_new = {}
    dimensions = {
        "结构型": ['PCI', 'R'],
        "环境型": ['AR', 'YSM'],
        "效率型": ['RT', 'AT', 'AADT']
    }
    for dim, cols in dimensions.items():
        scaled_new[dim] = scalers[dim].transform(new_data[cols])
    
    # 用历史PCA模型计算新主成分
    new_pc = {}
    for dim in dimensions.keys():
        new_pc[dim] = pca_models[dim].transform(scaled_new[dim]).flatten()
        # 结构型主成分方向调整（同步骤3）
        if dim == "结构型":
            new_pc[dim] = -new_pc[dim]
    
    # 计算新风险值S
    new_S = (
        weights['PC_结构型'] * new_pc['结构型'] +
        weights['PC_环境型'] * new_pc['环境型'] +
        weights['PC_效率型'] * new_pc['效率型']
    )
    
    # 对比更新前后优先级变化（简化：用历史阈值）
    # 历史阈值（从原结果中提取）
    original_thresholds = {
        '紧急': pc_df[pc_df['priority'] == '紧急']['S'].min(),
        '高': pc_df[pc_df['priority'] == '高']['S'].min(),
        '中': pc_df[pc_df['priority'] == '中']['S'].min()
    }
    # 新数据优先级
    new_priority = []
    for s in new_S:
        if s >= original_thresholds['紧急']:
            new_priority.append('紧急')
        elif s >= original_thresholds['高']:
            new_priority.append('高')
        elif s >= original_thresholds['中']:
            new_priority.append('中')
        else:
            new_priority.append('低')
    # 统计变化
    change_counts = pd.Series(new_priority).value_counts()
    print("更新后新数据优先级分布：")
    print(change_counts)
    return new_data, new_priority

# ----------------------
# 7. 结果可视化
# ----------------------
def visualize_results(pc_df):
    # 1. 优先级分布饼图
    plt.figure(figsize=(10, 6))
    pc_df['priority'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2'])
    plt.title('道路维护优先级分布')
    plt.ylabel('')
    plt.savefig('priority_distribution.png', dpi=300)
    plt.close()
    
    # 2. 主成分与S的相关性
    plt.figure(figsize=(12, 5))
    corr_cols = ['PC_结构型', 'PC_环境型', 'PC_效率型', 'S']
    sns.heatmap(pc_df[corr_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('主成分与综合风险值S的相关性')
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()
    print("\n可视化结果已保存为PNG图片")

# ----------------------
# 主函数：串联全流程
# ----------------------
if __name__ == "__main__":
    # 步骤1：加载数据
    df = load_processed_data()
    
    # 步骤2：分维度标准化
    scaled_data, dimensions, scalers = split_and_standardize(df)
    
    # 步骤3：分维度PCA
    pc_df, pca_models = pca_per_dimension(scaled_data, dimensions)
    
    # 步骤4：熵权法计算权重
    weights = entropy_weight(pc_df)
    
    # 步骤5：划分优先级
    pc_df = assign_priority(pc_df, weights)
    
    # 步骤6：动态更新模拟
    new_data, new_priority = dynamic_update_demo(df, scalers, pca_models, weights)
    
    # 步骤7：可视化
    visualize_results(pc_df)
    
    # 保存最终结果
    result_df = pd.concat([df.reset_index(drop=True), pc_df], axis=1)
    result_df.to_csv("DATA/road_maintenance_priority.csv", index=False)
    print("\n最终结果已保存至DATA/road_maintenance_priority.csv")
    print("包含字段：原始指标 + 主成分 + 综合风险值S + 优先级")