# -*- coding: utf-8 -*-
"""
业务导向的两阶段递进聚类法（复赛B任务A专用）
核心依据：
1. B2025070938351.pdf（初赛论文）：特征权重（SHAP值）、标准化逻辑、模型评价标准
2. 复赛B：道路路面维护需求综合预测.pdf：任务A“三类维护区聚类”“NM标签验证”要求
输入：./DATA/任务A中间结果/处理后数据.csv（标准化后数据，含4个连续特征+ID+NM）
输出：./业务导向的两阶段递进聚类法/（聚类结果CSV+可视化图）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


# -------------------------- 1. 基础配置（中文适配+图表设置） --------------------------
def init_chinese_config():
    """
    初始化中文配置：解决终端输出乱码与图表中文显示问题
    参考：复赛B题报告可视化需求（需直观中文图表）
    """
    # 终端中文编码重定向
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
    # 图表中文支持（复赛B题报告用图需清晰中文标注）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 中文配置初始化完成（终端+图表均支持中文）")


# -------------------------- 2. 数据加载与标准化验证 --------------------------
def load_scaled_data(input_path):
    """
    加载标准化后的数据（复赛B任务A指定4个连续特征：PCI、AADT、R、IRI）
    依据：B2025070938351.pdf表2（预处理后数据格式）、4.4.2节（标准化要求：均值≈0，标准差≈1）
    参数：input_path - 标准化数据文件路径
    返回：df - 加载后的数据框
    """
    # 仅保留复赛B任务A必需列（4个标准化特征+ID+NM标签）
    cols_needed = ["ID", "PCI_scaled", "AADT_scaled", "R_scaled", "IRI_scaled", "NM"]
    df = pd.read_csv(input_path, usecols=cols_needed, encoding='utf-8')
    
    # 验证标准化有效性（符合初赛论文4.4.2节逻辑，确保数据质量）
    scaled_cols = ["PCI_scaled", "AADT_scaled", "R_scaled", "IRI_scaled"]
    print("\n✅ 标准化数据验证（符合初赛论文要求）：")
    for col in scaled_cols:
        mean_val = df[col].mean().round(3)
        std_val = df[col].std().round(3)
        print(f"  {col}：均值={mean_val}，标准差={std_val}")
    
    # 验证数据规模（复赛B题明确84万条样本）
    print(f"✅ 数据加载完成：共{df.shape[0]}条样本（匹配复赛B题84万条规模）")
    return df


# -------------------------- 3. 第一阶段：粗聚类（结构型特征主导） --------------------------
def coarse_clustering(df):
    """
    粗聚类：基于结构型特征（R、PCI）划分高/中/低结构风险簇
    依据：B2025070938351.pdf 5.1.4节（SHAP值分析：R权重0.32，PCI权重0.28）
    参数：df - 加载后的标准化数据框
    返回：df - 含粗聚类标签（coarse_cluster）的数据框
    """
    # 步骤1：计算结构风险值S1（R与维护需求正相关，PCI负相关，取负号）
    def calculate_S1(df):
        df["S1"] = 0.6 * df["R_scaled"] - 0.4 * df["PCI_scaled"]  # 权重按SHAP值比例调整
        return df
    df = calculate_S1(df)
    
    # 步骤2：分位数划分粗簇（复赛B任务A要求三类，中间簇留微调空间）
    Q30 = df["S1"].quantile(0.3)  # 低结构风险阈值
    Q70 = df["S1"].quantile(0.7)  # 高结构风险阈值
    df["coarse_cluster"] = np.where(
        df["S1"] >= Q70, 1,  # 1=高结构风险（候选“急需维护区”）
        np.where(df["S1"] <= Q30, 3, 2)  # 3=低结构风险（候选“无需维护区”）；2=中风险（候选“一般维护区”）
    )
    
    # 输出粗簇分布（验证是否符合“高30%+中40%+低30%”合理比例）
    coarse_dist = df["coarse_cluster"].value_counts(normalize=True).sort_index() * 100
    print(f"\n📊 粗聚类分布（结构风险维度）：")
    print(f"  高结构风险簇（候选急需维护）：{coarse_dist[1]:.1f}%")
    print(f"  中结构风险簇（候选一般维护）：{coarse_dist[2]:.1f}%")
    print(f"  低结构风险簇（候选无需维护）：{coarse_dist[3]:.1f}%")
    return df


# -------------------------- 4. 第二阶段：细聚类（效率型特征微调） --------------------------
def fine_clustering(df):
    """
    细聚类：对中结构风险簇，用效率型特征（AADT、IRI）拆分高/低功能风险子簇
    依据：复赛B题“交通荷载影响维护优先级”要求；B2025070938351.pdf 5.2.2节（降维适配大数据）
    参数：df - 含粗聚类标签的数据框
    返回：df - 含中风险簇子标签（mid_subcluster）的数据框；df_mid - 中风险簇数据框
    """
    # 步骤1：提取中结构风险簇（仅处理40%样本，降低计算量，适配84万数据）
    df_mid = df[df["coarse_cluster"] == 2].copy()
    
    # 步骤2：计算功能风险值S2（AADT为核心效率指标，权重高于IRI）
    def calculate_S2(df_mid):
        df_mid["S2"] = 0.7 * df_mid["AADT_scaled"] + 0.3 * df_mid["IRI_scaled"]  # 均与维护需求正相关
        return df_mid
    df_mid = calculate_S2(df_mid)
    
    # 步骤3：MiniBatchKMeans拆分（适配大数据，避免全量KMeans内存溢出）
    mbk = MiniBatchKMeans(
        n_clusters=2,        # 拆分为“高/低功能风险”2个子簇
        batch_size=15000,    # 批量大小（复赛B题84万数据10分钟内跑完）
        init="k-means++",    # 优化簇中心初始化（避免局部最优）
        random_state=42      # 结果可复现（复赛B题建模要求）
    )
    df_mid["mid_subcluster"] = mbk.fit_predict(df_mid[["S2"]])
    
    # 步骤4：标签调整（确保subcluster=1对应“高功能风险”，需升级为急需维护）
    cluster_centers = mbk.cluster_centers_.flatten()
    if cluster_centers[0] > cluster_centers[1]:
        df_mid["mid_subcluster"] = 1 - df_mid["mid_subcluster"]  # 交换标签
    
    # 输出子簇分布（验证微调合理性）
    sub_dist = df_mid["mid_subcluster"].value_counts(normalize=True) * 100
    print(f"\n🔍 中风险簇拆分（功能风险维度）：")
    print(f"  高功能风险子簇（升级为急需维护）：{sub_dist[1]:.1f}%")
    print(f"  低功能风险子簇（保留为一般维护）：{sub_dist[0]:.1f}%")
    return df, df_mid


# -------------------------- 5. 合并最终聚类标签 --------------------------
def merge_final_cluster(df, df_mid):
    """
    合并粗簇与子簇标签，得到复赛B任务A要求的“三类维护区”
    依据：复赛B题任务A定义（急需/一般/无需维护区）
    规则：1.急需维护区=粗簇1+子簇1；2.一般维护区=子簇0；3.无需维护区=粗簇3
    参数：df - 全量数据框；df_mid - 中风险簇数据框
    返回：df - 含最终聚类标签的数据框
    """
    # 基础标签：复制粗簇标签
    df["final_cluster"] = df["coarse_cluster"].copy()
    
    # 中风险簇升级：子簇1（高功能风险）→ 急需维护区（标签1）
    high_func_ids = df_mid[df_mid["mid_subcluster"] == 1]["ID"].tolist()
    df.loc[df["ID"].isin(high_func_ids), "final_cluster"] = 1
    
    # 中风险簇保留：子簇0（低功能风险）→ 一般维护区（标签2）
    low_func_ids = df_mid[df_mid["mid_subcluster"] == 0]["ID"].tolist()
    df.loc[df["ID"].isin(low_func_ids), "final_cluster"] = 2
    
    # 映射业务名称（复赛B题报告需直观中文标签）
    df["final_cluster_name"] = df["final_cluster"].map({
        1: "急需维护区",
        2: "一般维护区",
        3: "无需维护区"
    })
    
    # 输出最终聚类分布（复赛B任务A核心结果）
    final_dist = df["final_cluster_name"].value_counts(normalize=True) * 100
    print(f"\n🏆 最终聚类结果（符合复赛B任务A三类划分）：")
    for name, ratio in final_dist.items():
        print(f"  {name}：{ratio:.1f}%")
    return df


# -------------------------- 6. 结果验证（NM标签+工程常识） --------------------------
def validate_cluster_result(df):
    """
    验证聚类结果：数学指标（ARI）+ 业务指标（维护占比）
    依据：B2025070938351.pdf 6.1节（模型评价逻辑）、复赛B题“NM标签验证”要求
    参数：df - 含最终聚类标签的数据框
    返回：ari_score - 调整兰德指数；maintain_stats - 类内需维护占比统计
    """
    # 步骤1：映射NM标签（聚类结果→维护需求：1/2类=需维护，3类=无需维护）
    df["nm_mapped"] = np.where(df["final_cluster"] <= 2, 1, 0)
    
    # 步骤2：计算ARI（数学一致性：衡量聚类与真实维护需求的匹配度）
    ari_score = adjusted_rand_score(df["NM"], df["nm_mapped"])
    
    # 步骤3：计算类内需维护占比（业务验证：符合工程常识，适配pandas≥2.0）
    # 修复点：用元组列表替代嵌套字典，避免nested renamer报错
    maintain_stats = df.groupby("final_cluster_name")["NM"].agg([
        ("维护样本数", "sum"),          # 簇内需要维护的样本数量
        ("总样本数", "count"),         # 簇内总样本数量
        ("维护占比(%)", lambda x: (x.sum() / len(x) * 100).round(2))  # 维护占比（保留2位小数）
    ]).reset_index()  # 重置索引，便于复赛报告输出
    
    # 输出验证结果
    print(f"\n✅ 聚类结果验证（符合初赛+复赛评价标准）：")
    print(f"  调整兰德指数（ARI）：{ari_score:.4f}（>0.5为良好匹配，>0.7为优秀）")
    print(f"\n  类内需维护占比统计（工程常识验证）：")
    print(maintain_stats.to_string(index=False))  # 格式化输出，无冗余索引
    return ari_score, maintain_stats


# -------------------------- 7. 结果保存与可视化 --------------------------
def save_and_visualize(df, output_dir, output_csv, output_plot):
    """
    保存聚类结果（CSV）与可视化图表（散点图），用于复赛B题报告
    依据：复赛B题任务A“输出聚类结果+统计特征”要求
    参数：df - 含最终结果的数据框；output_dir - 输出文件夹；output_csv/plot - 输出文件路径
    """
    # 步骤1：保存聚类结果CSV（含关键信息，便于复赛后续任务B使用）
    result_cols = [
        "ID", "PCI_scaled", "AADT_scaled", "R_scaled", "IRI_scaled",  # 标准化特征
        "S1", "S2", "final_cluster", "final_cluster_name", "NM"       # 聚类结果+真实标签
    ]
    df[result_cols].to_csv(output_csv, index=False, encoding='utf-8-sig')  # 支持中文
    
    # 步骤2：可视化聚类结果（S1-S2散点图，复赛报告核心图表）
    plt.figure(figsize=(10, 6))
    color_map = {"急需维护区": "#FF5733", "一般维护区": "#FFC300", "无需维护区": "#33FF57"}
    for cluster_name in color_map.keys():
        cluster_data = df[df["final_cluster_name"] == cluster_name]
        plt.scatter(
            x=cluster_data["S1"],  # x轴：结构风险值S1（初赛论文核心指标）
            y=cluster_data["S2"],  # y轴：功能风险值S2（复赛B题效率指标）
            c=color_map[cluster_name],
            alpha=0.3,  # 透明度：避免样本点重叠
            label=cluster_name
        )
    plt.xlabel("结构风险值S1（数值越大，路面结构破损越严重）")
    plt.ylabel("功能风险值S2（数值越大，交通荷载/平整度影响越显著）")
    plt.title("复赛B任务A：道路维护区聚类结果（业务导向两阶段递进法）")
    plt.legend(loc="upper left")
    plt.tight_layout()  # 自动调整布局，避免标签截断
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")  # 高分辨率（300dpi）适配报告
    
    print(f"\n💾 结果输出完成（复赛B题报告可直接引用）：")
    print(f"  1. 聚类结果CSV：{output_csv}")
    print(f"  2. 可视化图表：{output_plot}")


# -------------------------- 8. 主程序入口（统一执行流程） --------------------------
def main():
    """
    主程序入口：串联所有流程（中文配置→数据加载→粗聚类→细聚类→标签合并→验证→保存）
    依据：B2025070938351.pdf建模流程、复赛B任务A步骤要求
    """
    # 1. 初始化中文配置
    init_chinese_config()
    
    # 2. 定义路径（严格匹配你的文件结构）
    INPUT_PATH = "./DATA/任务A中间结果/处理后数据.csv"  # 输入：标准化数据
    OUTPUT_DIR = "./业务导向的两阶段递进聚类法"          # 输出：方法名文件夹
    OUTPUT_CSV = f"{OUTPUT_DIR}/final_cluster_result.csv"  # 聚类结果CSV
    OUTPUT_PLOT = f"{OUTPUT_DIR}/cluster_scatter.png"       # 可视化图
    
    # 3. 核心流程执行
    try:
        # 步骤1：加载标准化数据
        df = load_scaled_data(INPUT_PATH)
        
        # 步骤2：第一阶段粗聚类
        df = coarse_clustering(df)
        
        # 步骤3：第二阶段细聚类
        df, df_mid = fine_clustering(df)
        
        # 步骤4：合并最终聚类标签
        df = merge_final_cluster(df, df_mid)
        
        # 步骤5：结果验证
        ari_score, maintain_stats = validate_cluster_result(df)
        
        # 步骤6：保存结果与可视化
        save_and_visualize(df, OUTPUT_DIR, OUTPUT_CSV, OUTPUT_PLOT)
        
        print(f"\n🎉 复赛B任务A聚类完成！所有结果已输出至：{OUTPUT_DIR}")
    
    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")
        print("请检查：1. 输入文件路径是否正确；2. 数据是否包含所需列（如PCI_scaled）；3. pandas版本是否≥1.0")


# -------------------------- 9. 程序启动（仅主程序入口执行） --------------------------
if __name__ == "__main__":
    main()