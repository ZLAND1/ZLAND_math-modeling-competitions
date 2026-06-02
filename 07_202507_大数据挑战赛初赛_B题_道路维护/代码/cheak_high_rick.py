# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr  # 用于二元变量相关性分析
from sklearn.model_selection import train_test_split

# 解决终端中文显示乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def detect_data_leakage(df, target_col='NM', feature_cols=None, random_state=42):
    """
    检测数据集中的数据泄露和强规则问题
    
    参数:
    df: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名（默认'NM'）
    feature_cols: 特征列名列表（None则自动排除目标列）
    random_state: 随机种子，确保结果可复现
    """
    print("===== 数据泄露与强规则检测开始 =====")
    print(f"数据集规模: {df.shape[0]}行, {df.shape[1]}列")
    print(f"目标变量: {target_col} (1=需要维护, 0=不需要)")
    print(f"目标变量分布: {df[target_col].value_counts(normalize=True).round(4) * 100}%\n")
    
    # 1. 初始化特征列
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col and col != 'ID']
    print(f"待检测特征: {feature_cols}\n")
    
    # 2. 检测特征与目标变量的强相关性
    print("===== 1. 特征与目标变量的相关性分析 =====")
    corr_results = []
    for col in feature_cols:
        # 计算点二列相关系数（适用于二元目标变量）
        corr, p_value = pointbiserialr(df[col], df[target_col])
        corr_results.append({
            '特征': col,
            '相关系数': abs(corr),  # 取绝对值，关注强度
            'p值': p_value,
            '相关性强度': '强' if abs(corr) > 0.8 else 
                         '中' if abs(corr) > 0.5 else '弱'
        })
    
    corr_df = pd.DataFrame(corr_results).sort_values(by='相关系数', ascending=False)
    print(corr_df)
    
    # 标记高风险特征（相关系数>0.8）
    high_risk = corr_df[corr_df['相关性强度'] == '强']
    if not high_risk.empty:
        print(f"\n警告: 发现{len(high_risk)}个高风险特征（相关系数>0.8），可能存在强规则:")
        print(high_risk['特征'].tolist())
    else:
        print("\n未发现强相关特征（相关系数>0.8）")
    
    # 3. 检测特征是否能单独完美预测目标变量（强规则）
    print("\n===== 2. 强规则检测（单一特征完美预测） =====")
    rule_threshold = 0.95  # 预测准确率超过此值视为强规则
    strong_rules = []
    
    for col in feature_cols:
        # 对连续特征进行分箱分析（找到最佳分割点）
        if df[col].nunique() > 10:  # 连续特征
            # 尝试不同分位数作为分割点
            thresholds = np.percentile(df[col].dropna(), [10, 25, 50, 75, 90])
            for thresh in thresholds:
                # 两种分割方向（>阈值和<阈值）
                for op in ['gt', 'lt']:
                    if op == 'gt':
                        pred = (df[col] > thresh).astype(int)
                    else:
                        pred = (df[col] < thresh).astype(int)
                    
                    accuracy = (pred == df[target_col]).mean()
                    if accuracy >= rule_threshold:
                        strong_rules.append({
                            '特征': col,
                            '规则': f"{col}{'>' if op=='gt' else '<'}{thresh:.2f}",
                            '准确率': accuracy
                        })
        else:  # 分类特征
            # 直接计算每个类别对应的目标变量比例
            group_stats = df.groupby(col)[target_col].agg(['mean', 'count'])
            # 检查是否存在某个类别中目标变量几乎全为0或1
            for val in group_stats.index:
                ratio = group_stats.loc[val, 'mean']
                if ratio > rule_threshold or ratio < (1 - rule_threshold):
                    strong_rules.append({
                        '特征': col,
                        '规则': f"{col}={val}时, {target_col}={'几乎全为1' if ratio>rule_threshold else '几乎全为0'}",
                        '准确率': max(ratio, 1 - ratio)
                    })
    
    if strong_rules:
        print(f"发现{len(strong_rules)}个潜在强规则（准确率>=95%）:")
        for rule in strong_rules:
            print(f"特征: {rule['特征']}, 规则: {rule['规则']}, 准确率: {rule['准确率']:.4f}")
    else:
        print("未发现强规则（单一特征预测准确率均<95%）")
    
    # 4. 检测训练集与测试集分布一致性（避免数据泄露）
    print("\n===== 3. 训练集与测试集分布一致性检测 =====")
    # 分割数据集
    train, test = train_test_split(df, test_size=0.3, random_state=random_state, stratify=df[target_col])
    
    # 比较特征分布差异
    distribution_diff = []
    for col in feature_cols:
        # 计算KS统计量（衡量分布差异）
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()
        
        # 合并所有值并排序
        all_vals = np.sort(np.unique(np.concatenate([train_vals, test_vals])))
        if len(all_vals) < 2:
            ks_stat = 0.0  # 特征值唯一，无差异
        else:
            # 计算累积分布函数
            cdf_train = np.array([np.mean(train_vals <= x) for x in all_vals])
            cdf_test = np.array([np.mean(test_vals <= x) for x in all_vals])
            ks_stat = np.max(np.abs(cdf_train - cdf_test))
        
        distribution_diff.append({
            '特征': col,
            'KS统计量': ks_stat,
            '分布一致性': '差' if ks_stat > 0.2 else 
                         '中' if ks_stat > 0.1 else '好'
        })
    
    ks_df = pd.DataFrame(distribution_diff).sort_values(by='KS统计量', ascending=False)
    print(ks_df)
    
    # 标记分布差异大的特征
    bad_dist = ks_df[ks_df['分布一致性'] == '差']
    if not bad_dist.empty:
        print(f"\n警告: 发现{len(bad_dist)}个特征在训练/测试集分布差异大（KS>0.2），可能存在数据泄露:")
        print(bad_dist['特征'].tolist())
    else:
        print("\n所有特征在训练/测试集分布一致性良好（KS<=0.2）")
    
    # 5. 可视化核心风险特征
    print("\n===== 4. 核心风险特征可视化 =====")
    # 选择相关性最高的2个特征进行可视化
    top_features = corr_df['特征'].head(2).tolist()
    for col in top_features:
        plt.figure(figsize=(10, 4))
        
        # 连续特征绘制箱线图
        if df[col].nunique() > 10:
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f'{col}与{target_col}的关系（箱线图）')
            # 添加分割线（如果存在强规则）
            for rule in strong_rules:
                if rule['特征'] == col and ('>' in rule['规则'] or '<' in rule['规则']):
                    thresh = float(rule['规则'].split('>')[-1].split('<')[-1])
                    plt.axhline(y=thresh, color='r', linestyle='--', 
                               label=f'强规则阈值: {thresh:.2f}')
            plt.legend()
        
        # 分类特征绘制柱状图
        else:
            count_df = df.groupby([col, target_col]).size().unstack()
            count_df = count_df.div(count_df.sum(axis=1), axis=0)  # 计算比例
            count_df.plot(kind='bar', stacked=True)
            plt.title(f'{col}与{target_col}的关系（比例堆叠图）')
            plt.ylabel(f'{target_col}=1的比例')
        
        plt.tight_layout()
        plt.show()
    
    print("\n===== 检测完成 =====")
    print("总结建议:")
    if not high_risk.empty or strong_rules or not bad_dist.empty:
        print("- 存在潜在数据问题，请重点检查上述标记的高风险特征和强规则")
        print("- 确认特征是否直接决定了目标变量（如PCI低于某个值必需要维护）")
        print("- 检查训练/测试集划分是否合理，避免数据泄露")
    else:
        print("- 未发现明显的数据泄露或强规则问题，模型结果可信度较高")

# 使用示例
if __name__ == "__main__":
    # 加载数据（替换为你的数据路径）
    data_path = "DATA/processed_road_data.csv"
    df = pd.read_csv(data_path)
    
    # 定义特征列（根据实际情况调整）
    feature_columns = ['PCI', 'RT', 'AADT', 'AT', 'AR', 'R', 'IRI', 'YSM']
    
    # 执行检测
    detect_data_leakage(
        df=df,
        target_col='NM',
        feature_cols=feature_columns
    )
