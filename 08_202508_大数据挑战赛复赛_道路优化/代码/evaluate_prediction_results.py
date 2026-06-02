import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, cohen_kappa_score)

# 忽略警告
warnings.filterwarnings('ignore')

# 解决终端中文显示乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def calculate_evaluation_metrics(result_path):
    """
    从预测结果文件计算并返回评估指标
    参数:
        result_path: 预测结果CSV文件路径
    返回:
        metrics_df: 包含评估指标的DataFrame
    """
    print("="*50)
    print("开始计算预测评估指标")
    print("="*50)
    
    # 检查文件是否存在
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"预测结果文件不存在: {result_path}")
    
    # 读取预测结果
    try:
        result_df = pd.read_csv(result_path, encoding="utf-8-sig")
        print(f"✅ 成功读取预测结果：{result_df.shape[0]}行数据")
    except Exception as e:
        raise Exception(f"读取文件失败：{str(e)}")
    
    # 检查必要的列是否存在
    required_cols = ["NM", "y_hat_i", "y_hat_prob"]
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        raise ValueError(f"预测结果文件缺少必要的列：{missing_cols}")
    
    # 提取真实标签、预测标签和预测概率
    y_true = result_df["NM"].values       # 真实标签
    y_pred = result_df["y_hat_i"].values  # 预测标签
    y_prob = result_df["y_hat_prob"].values  # 预测概率
    
    # 计算评估指标
    print("\n正在计算评估指标...")
    metrics = {
        "指标名称": ["准确率(Accuracy)", "精确率(Precision)", "召回率(Recall)", 
                   "F1-Score", "AUC-ROC", "Kappa系数"],
        "计算值": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, pos_label=1),
            recall_score(y_true, y_pred, pos_label=1),
            f1_score(y_true, y_pred, pos_label=1),
            roc_auc_score(y_true, y_prob),
            cohen_kappa_score(y_true, y_pred)
        ],
        "论文参考目标值": ["≥0.85", "≥0.80", "≥0.85", "≥0.82", "≥0.90", "≥0.75"]
    }
    
    # 整理结果
    metrics_df = pd.DataFrame(metrics)
    metrics_df["计算值"] = metrics_df["计算值"].round(4)  # 保留4位小数
    
    # 评估是否达到目标
    metrics_df["是否达标"] = [
        "是" if metrics_df["计算值"][i] >= float(metrics_df["论文参考目标值"][i][2:]) else "否"
        for i in range(len(metrics_df))
    ]
    
    return metrics_df

def save_and_display_metrics(metrics_df, output_dir):
    """保存并显示评估指标"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存评估结果
    eval_path = os.path.join(output_dir, "预测性能评估详细结果.csv")
    metrics_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 评估结果已保存至：{eval_path}")
    
    # 显示评估结果
    print("\n" + "="*50)
    print("预测性能评估结果（与论文目标对比）")
    print("="*50)
    print(metrics_df.to_string(index=False))
    
    # 总体评估
    if all(metrics_df["是否达标"] == "是"):
        print("\n🎉 所有指标均达到论文参考目标值！")
    else:
        not达标 = metrics_df[metrics_df["是否达标"] == "否"]["指标名称"].tolist()
        print(f"\n⚠️ 以下指标未达到论文参考目标值：{not达标}")

if __name__ == "__main__":
    # 预测结果文件路径（请根据实际输出路径修改）
    result_path = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/任务B预测结果/维护需求预测结果.csv"
    output_dir = os.path.dirname(result_path)  # 与预测结果同目录
    
    try:
        # 计算评估指标
        metrics_df = calculate_evaluation_metrics(result_path)
        
        # 保存并显示结果
        save_and_display_metrics(metrics_df, output_dir)
        
    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")
