import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb

# 加载数据
processed_df = pd.read_csv("DATA\processed_road_data.csv")
X = processed_df.drop(['ID', 'NM'], axis=1)
y = processed_df['NM']

# 处理缺失值
numeric_features = ['PCI', 'AADT', 'AR', 'R', 'IRI', 'YSM']
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
categorical_features = ['RT', 'AT']
X[categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 只选择数值型特征进行多项式转换（分类型特征保持不变）
numeric_X_train = X_train[numeric_features]
numeric_X_test = X_test[numeric_features]
categorical_X_train = X_train[categorical_features]
categorical_X_test = X_test[categorical_features]

# 评估函数
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name}模型评估:")
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(f"召回率: {recall_score(y_true, y_pred):.4f}")
    print(f"F1值: {f1_score(y_true, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

# 1. 多项式回归模型（带正则化）
# 使用管道组合多项式特征和Ridge回归（解决过拟合）
poly_degree = 2  # 尝试2阶多项式（高阶可能导致过拟合）
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 标准化
    ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),  # 生成多项式特征
    ('ridge', Ridge(alpha=10.0))  # 带L2正则化的线性回归
])

# 训练多项式回归模型
poly_pipeline.fit(numeric_X_train, y_train)

# 预测（将概率转换为二分类）
y_pred_poly = (poly_pipeline.predict(numeric_X_test) >= 0.5).astype(int)

# 评估多项式回归
poly_results = evaluate_model(y_test, y_pred_poly, f"{poly_degree}阶多项式回归（Ridge正则化）")

# 查看生成的多项式特征（理解模型学习的特征交互）
poly_features = poly_pipeline.named_steps['poly'].get_feature_names_out(numeric_features)
print(f"\n生成的多项式特征（共{len(poly_features)}个）:")
print(poly_features[:10])  # 显示前10个特征

# 2. 对比其他适用模型

# 2.1 带正则化的逻辑回归（处理强相关性特征）
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', Ridge(alpha=5.0))  # 使用Ridge作为分类器（等价于带L2正则的逻辑回归）
])
logistic_pipeline.fit(numeric_X_train, y_train)
y_pred_logistic = (logistic_pipeline.predict(numeric_X_test) >= 0.5).astype(int)
logistic_results = evaluate_model(y_test, y_pred_logistic, "带L2正则的逻辑回归")

# 2.2 随机森林（抗过拟合，处理非线性关系）
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,  # 限制树深度，避免过拟合强规则
    min_samples_split=50,  # 增加分裂所需样本数
    random_state=42,
    class_weight='balanced'
)
# 合并数值和分类型特征
rf_X_train = pd.concat([numeric_X_train, categorical_X_train], axis=1)
rf_X_test = pd.concat([numeric_X_test, categorical_X_test], axis=1)
rf_model.fit(rf_X_train, y_train)
y_pred_rf = rf_model.predict(rf_X_test)
rf_results = evaluate_model(y_test, y_pred_rf, "随机森林（限制复杂度）")

# 2.3 梯度提升树（XGBoost带更强正则化）
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=4,  # 减小树深度
    learning_rate=0.1,
    reg_alpha=5.0,  # L1正则
    reg_lambda=10.0,  # L2正则
    scale_pos_weight=sum(y_train==0)/sum(y_train==1),
    random_state=42
)
xgb_model.fit(rf_X_train, y_train)
y_pred_xgb = xgb_model.predict(rf_X_test)
xgb_results = evaluate_model(y_test, y_pred_xgb, "带强正则的XGBoost")

# 3. 模型结果对比
results = pd.DataFrame([poly_results, logistic_results, rf_results, xgb_results])
results = results.sort_values(by='f1', ascending=False)
print("\n各模型性能对比:")
print(results)

# 4. 多项式回归特征重要性分析（基于系数绝对值）
poly_coef = poly_pipeline.named_steps['ridge'].coef_
poly_importance = pd.DataFrame({
    '特征': poly_features,
    '重要性': np.abs(poly_coef)
}).sort_values(by='重要性', ascending=False)

print("\n多项式回归特征重要性（前10名）:")
print(poly_importance.head(10))

# 可视化多项式特征与目标变量的关系（以R为例）
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['R'], y=y_test, alpha=0.5, label='实际值')
# 生成预测曲线
r_range = np.linspace(X_test['R'].min(), X_test['R'].max(), 100).reshape(-1, 1)
# 构造其他特征的中位数作为参考
other_features = numeric_X_test.drop('R', axis=1).median().values
# 生成预测数据
pred_data = np.repeat(other_features.reshape(1, -1), 100, axis=0)
pred_data = np.hstack([pred_data[:, :2], r_range, pred_data[:, 2:]])  # 插入R的取值
# 标准化并预测
pred_data_scaled = poly_pipeline.named_steps['scaler'].transform(pred_data)
y_pred_curve = poly_pipeline.named_steps['ridge'].predict(
    poly_pipeline.named_steps['poly'].transform(pred_data_scaled)
)
sns.lineplot(x=r_range.flatten(), y=y_pred_curve, color='red', label=f'{poly_degree}阶多项式预测')
plt.xlabel('车辙深度（R）')
plt.ylabel('需要维护的概率（NM=1）')
plt.title('多项式回归捕捉的非线性关系')
plt.legend()
plt.show()
    