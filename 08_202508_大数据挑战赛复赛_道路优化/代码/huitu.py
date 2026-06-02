import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建输出目录
output_dir = "聚类结果可视化图表"
os.makedirs(output_dir, exist_ok=True)

def generate_sample_data(n_samples=3000):
    """生成符合描述特征的模拟数据"""
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    
    # 1. 急需维护区（红色点）：PC1负方向，PC2正方向，PCI低
    n_urgent = int(n_samples * 0.3)
    urgent_data = {
        'PC1': np.random.normal(-3, 1, n_urgent),
        'PC2': np.random.normal(2, 1, n_urgent),
        'PCI': np.random.uniform(10, 40, n_urgent),
        'cluster': '急需维护区'
    }
    
    # 2. 无需维护区（蓝色点）：PC1正方向，PC2负方向，PCI高
    n_no_need = int(n_samples * 0.4)
    no_need_data = {
        'PC1': np.random.normal(3, 1, n_no_need),
        'PC2': np.random.normal(-2, 1, n_no_need),
        'PCI': np.random.uniform(80, 100, n_no_need),
        'cluster': '无需维护区'
    }
    
    # 3. 一般维护区（绿色点）：中间分布，PCI中等
    n_general = n_samples - n_urgent - n_no_need
    general_data = {
        'PC1': np.random.normal(0, 1, n_general),
        'PC2': np.random.normal(0, 1, n_general),
        'PCI': np.random.uniform(50, 80, n_general),
        'cluster': '一般维护区'
    }
    
    # 合并数据
    df_urgent = pd.DataFrame(urgent_data)
    df_no_need = pd.DataFrame(no_need_data)
    df_general = pd.DataFrame(general_data)
    
    return pd.concat([df_urgent, df_no_need, df_general], ignore_index=True)

def plot_pca_scatter(df):
    """绘制PCA降维后的聚类散点图"""
    plt.figure(figsize=(12, 8))
    
    # 定义颜色和标记
    colors = {'急需维护区': 'red', '无需维护区': 'blue', '一般维护区': 'green'}
    markers = {'急需维护区': 'o', '无需维护区': 's', '一般维护区': '^'}
    
    # 绘制散点图
    for cluster in df['cluster'].unique():
        subset = df[df['cluster'] == cluster]
        plt.scatter(
            subset['PC1'], 
            subset['PC2'], 
            c=colors[cluster], 
            label=cluster,
            marker=markers[cluster],
            alpha=0.6,
            s=50
        )
    
    # 添加标题和标签
    plt.title('图5-1：PCA降维后的聚类散点图', fontsize=16)
    plt.xlabel(f'PC1（解释方差68.2%）', fontsize=12)
    plt.ylabel(f'PC2（解释方差28.1%）', fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(title='维护区域类型', fontsize=10, title_fontsize=12)
    
    # 添加解释文本
    plt.figtext(0.5, 0.01, 
                'PC1：主要负载PCI（正相关）与Rutting（负相关）；PC2：主要负载AADT（正相关）与IRI（正相关）', 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    save_path = os.path.join(output_dir, 'PCA聚类散点图.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCA聚类散点图已保存至：{save_path}")
    
    plt.show()

def plot_pci_boxplot(df):
    """绘制三类维护区的PCI箱线图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制箱线图
    sns.boxplot(x='cluster', y='PCI', data=df, 
                order=['无需维护区', '一般维护区', '急需维护区'],
                palette={'急需维护区': 'red', '无需维护区': 'blue', '一般维护区': 'green'})
    
    # 添加散点显示分布
    sns.stripplot(x='cluster', y='PCI', data=df, 
                  order=['无需维护区', '一般维护区', '急需维护区'],
                  color='black', alpha=0.3, size=2)
    
    # 添加标题和标签
    plt.title('图5-2：三类维护区的PCI箱线图', fontsize=16)
    plt.xlabel('维护区域类型', fontsize=12)
    plt.ylabel('PCI值', fontsize=12)
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 设置y轴范围
    plt.ylim(0, 110)
    
    # 添加解释文本
    plt.figtext(0.5, 0.01, 
                'PCI值越高，路面结构状态越好', 
                ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    save_path = os.path.join(output_dir, 'PCI箱线图.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCI箱线图已保存至：{save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 生成模拟数据（实际使用时可替换为真实数据）
    print("生成模拟数据...")
    df = generate_sample_data(n_samples=3000)
    
    # 显示数据基本信息
    print("\n数据基本信息：")
    print(df.groupby('cluster').agg({
        'PC1': ['mean', 'std'],
        'PC2': ['mean', 'std'],
        'PCI': ['mean', 'std', 'count']
    }))
    
    # 绘制PCA聚类散点图
    plot_pca_scatter(df)
    
    # 绘制PCI箱线图
    plot_pci_boxplot(df)
    
    print("\n所有图表绘制完成！")
    