# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import xgboost as xgb
import shap

# 解决终端中文显示乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ----------------------
# 步骤1：数据加载与预处理
# ----------------------
processed_df = pd.read_csv("DATA/processed_road_data.csv")  # 输入前期处理后的文件
X = processed_df.drop(['ID', 'NM'], axis=1)  # 特征
y = processed_df['NM']  # 目标变量

# 处理缺失值
numeric_features = ['PCI', 'AADT', 'AR', 'R', 'IRI', 'YSM']
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
categorical_features = ['RT', 'AT']
X[categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化（供逻辑回归用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# 步骤2：模型训练与评估
# ----------------------
# XGBoost模型
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', n_estimators=100, max_depth=5, learning_rate=0.1,
    scale_pos_weight=sum(y_train==0)/sum(y_train==1)
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# 输出评估指标
print("XGBoost模型评估：")
print(f"准确率：{accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"召回率：{recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1值：{f1_score(y_test, y_pred_xgb):.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred_xgb))

# ----------------------
# 步骤3：特征重要性分析
# ----------------------
# SHAP全局重要性
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type='bar')