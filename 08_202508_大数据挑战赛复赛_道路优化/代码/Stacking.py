import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import io
import warnings
warnings.filterwarnings('ignore')


# 导入依赖库（指定文件提及的模型库）
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, cohen_kappa_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import shap  # B2025070938351.pdf要求SHAP特征分析

def safe_redirect():
    """安全处理终端编码，避免强制重定向导致崩溃"""
    try:
        current_enc = sys.stdout.encoding.lower() if sys.stdout.encoding else 'unknown'
        if current_enc not in ['utf-8', 'utf8']:
            # 行缓冲模式：输出立即刷新，减少编码冲突
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                line_buffering=True
            )
            print(f"✅ 终端编码重定向完成（原编码：{current_enc} → 目标编码：utf-8）")
        else:
            print(f"✅ 终端已支持UTF-8编码（当前编码：{current_enc}），无需重定向")
    except Exception as e:
        print(f"⚠️  编码重定向失败（不影响核心功能），错误：{str(e)[:50]}")

# -------------------------- 1. 配置参数（严格匹配指定文件） --------------------------
# 输入数据：已完成初步预处理（含RT、AT、YSM列，无缺失值）
INPUT_PREPROCESSED_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/processed_road_data.csv"
OUTPUT_PRED_PATH = r"D:/数学建模比赛/大数据挑战赛复赛2025826_829/DATA/任务B预测结果"
os.makedirs(OUTPUT_PRED_PATH, exist_ok=True)

# 特征与标签（初步预处理后的列名，B2025070938351.pdf表2）
CORE_FEATURES = ["PCI", "AADT", "RT", "AT", "YSM", "AR", "R", "IRI"]  # AR=Average Rainfall, R=Rutting
LABEL_COL = "NM"  # Needs Maintenance→NM（初步预处理已简化）
ID_COL = "ID"     # Segment ID→ID（初步预处理已简化）

# 模型超参（第二问.pdf表1-14）
XGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 100, 
              "random_state": 42, "objective": "binary:logistic"}
CAT_PARAMS = {"random_state": 42, "learning_rate": 0.05, "depth": 5, "iterations": 100,
              "cat_features": ["RT", "AT"]}  # 直接用初步预处理的量化分类特征
LGB_PARAMS = {"num_leaves": 31, "subsample": 0.8, "learning_rate": 0.05, 
              "n_estimators": 100, "random_state": 42, "objective": "binary"}
MLP_PARAMS = {
    "hidden_layer_sizes": (12, 6), 
    "activation": "logistic",  # 修复：'logistic'对应sklearn中的sigmoid激活，符合文件要求
    "solver": "adam", 
    "random_state": 42, 
    "max_iter": 200
}
LR_PARAMS = {"C": 0.1, "penalty": "l1", "solver": "liblinear", 
             "class_weight": "balanced", "random_state": 42}

# -------------------------- 2. 数据加载与初步预处理校验（衔接论文） --------------------------
def load_and_validate_preprocessed_data(data_path):
    """
    核心：验证论文初步预处理结果，避免重复处理
    校验项：1. 必要列存在；2. 无缺失值；3. 分类特征已量化；4. YSM计算正确
    """
    print("="*50)
    print("步骤1：加载并校验论文初步预处理数据")
    print("="*50)
    
    # 加载数据
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ 初步预处理数据加载成功：{df.shape[0]}行 × {df.shape[1]}列")
    
    # 校验1：必要列是否存在（初步预处理后应包含的列）
    required_cols = [ID_COL, LABEL_COL] + CORE_FEATURES
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 初步预处理数据缺失列：{missing_cols}（参考B2025070938351.pdf表2）")
        raise SystemExit(1)
    
    # 校验2：无缺失值（论文4.3节要求）
    missing_values = df[CORE_FEATURES + [LABEL_COL]].isnull().sum()
    if missing_values.any():
        print(f"⚠️  初步预处理后仍有缺失值（论文要求无缺失），用分组中位数补充：")
        for col in CORE_FEATURES:
            if missing_values[col] > 0:
                median_val = df.groupby("RT")[col].median()  # 按道路类型分组（第二问.pdf要求）
                df[col] = df.groupby("RT")[col].transform(lambda x: x.fillna(median_val[x.name]))
        print("✅ 缺失值补充完成")
    else:
        print("✅ 无缺失值，符合论文4.3节数据质量要求")
    
    # 校验3：分类特征已量化（B2025070938351.pdf4.4.1节）
    rt_valid = df["RT"].isin([1, 2, 3]).all()
    at_valid = df["AT"].isin([1, 2]).all()
    if not rt_valid:
        print(f"❌ 道路类型RT未量化（论文要求1=支路、2=次干道、3=主干道）")
        raise SystemExit(1)
    if not at_valid:
        print(f"❌ 沥青类型AT未量化（论文要求1=混凝土、2=沥青）")
        raise SystemExit(1)
    print("✅ 分类特征已量化，符合论文预处理要求")
    
    # 校验4：YSM计算正确（2025 - 上次维护年份，范围0~20）
    if df["YSM"].min() < 0 or df["YSM"].max() > 20:
        print(f"⚠️ YSM值异常（范围{df['YSM'].min()}~{df['YSM'].max()}），论文要求0~20年，已裁剪")
        df["YSM"] = df["YSM"].clip(0, 20)

    # 新增：确保分类特征为整数类型（CatBoost要求）
    df["RT"] = df["RT"].astype(int)
    df["AT"] = df["AT"].astype(int)
    print("✅ 分类特征（RT/AT）已转为整数类型，符合CatBoost要求")
    
    return df

# -------------------------- 3. 补充预处理（论文初步处理后需新增的步骤） --------------------------
def supplement_preprocessing(df):
    """
    论文初步预处理未覆盖，需补充的操作（第二问.pdf1-9节特征衍生）
    1. 交通量等级（AADT_Level）；2. 路面综合损坏指数（Com_Damage）；3. 气候-交通交互特征（Rain_AADT）
    """
    print("\n" + "="*50)
    print("步骤2：补充论文要求的特征衍生（初步预处理未覆盖）")
    print("="*50)
    
    df = df.copy()
    
    # 1. 交通量等级（AADT_Level：低<5000、中5000-15000、高>15000，第二问.pdf1-9节）
    df["AADT_Level"] = pd.cut(df["AADT"], bins=[0, 5000, 15000, np.inf], labels=["Low", "Mid", "High"])
    df = pd.get_dummies(df, columns=["AADT_Level"], prefix="AADT_Level")
    
    # 2. 路面综合损坏指数（Com_Damage，第二问.pdf公式：(1-PCI/100)*0.5 + R/30*0.3 + IRI/10*0.2）
    df["Com_Damage"] = (1 - df["PCI"]/100)*0.5 + (df["R"]/30)*0.3 + (df["IRI"]/10)*0.2
    
    # 3. 气候-交通交互特征（Rain_AADT，第二问.pdf公式：AR * AADT / 10000）
    df["Rain_AADT"] = df["AR"] * df["AADT"] / 10000
    
    # 4. 沥青类型权重（Asphalt_Weight，第二问.pdf1-9节：SMA=0.2、开级配=0.1、致密=0，此处AT=2对应沥青，按类型映射）
    # 注：初步预处理AT=2为沥青，需确认细分类型（此处用论文默认映射）
    asphalt_weight_map = {1: 0, 2: 0.1}  # AT=1(混凝土)=0，AT=2(沥青默认开级配)=0.1
    df["Asphalt_Weight"] = df["AT"].map(asphalt_weight_map)
    
    # 标准化补充特征（第二问.pdf1-10节，仅连续特征）
    supplement_cont_feats = ["Com_Damage", "Rain_AADT"]
    scaler = StandardScaler()
    df[supplement_cont_feats] = scaler.fit_transform(df[supplement_cont_feats])
    
    # 最终模型输入特征（分模型适配）
    # XGBoost/LightGBM：量化分类+标准化连续
    model_features = (["PCI", "AADT", "YSM", "AR", "R", "IRI", "RT", "AT", "Asphalt_Weight", "Com_Damage", "Rain_AADT"] + 
                     [col for col in df.columns if col.startswith("AADT_Level_")])
    # CatBoost：保留原始量化分类（无需独热）
    cat_features = ["PCI", "AADT", "YSM", "AR", "R", "IRI", "RT", "AT", "Com_Damage", "Rain_AADT"]
    
    print(f"✅ 补充预处理完成，最终模型特征数：{len(model_features)}")
    return df, model_features, cat_features, scaler

# -------------------------- 4. Stacking集成预测（严格按第二问.pdf流程） --------------------------
def stacking_prediction(df, model_features, cat_features, label_col):
    """
    流程：5折分层CV→基模型输出概率→元模型融合（第二问.pdf1-12节）
    修复：CatBoost输入保留DataFrame，不转numpy数组
    """
    print("\n" + "="*50)
    print("步骤3：Stacking集成预测（第二问.pdf1-12节）")
    print("="*50)
    
    # 数据拆分（修复：X_cat保留DataFrame，不转.values）
    X = df[model_features].values  # XGBoost/LightGBM用numpy数组
    X_cat = df[cat_features]       # CatBoost用DataFrame（关键修复）
    y = df[label_col].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_preds = np.zeros((len(X), 4))  # 4个基模型概率输出
    
    # 第一层：训练基模型（第二问.pdf1-14节）
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"🔍 折{fold+1}/5 训练基模型...")
        # 修复：CatBoost输入用DataFrame的行索引切片，保留列名
        X_train_cat = X_cat.iloc[train_idx]  # 不转.values，保留DataFrame
        X_val_cat = X_cat.iloc[val_idx]      # 不转.values，保留DataFrame
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 1. XGBoost（量化+连续特征，用numpy数组）
        X_train = df.iloc[train_idx][model_features].values
        X_val = df.iloc[val_idx][model_features].values
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        base_preds[val_idx, 0] = xgb_model.predict_proba(X_val)[:, 1]
        
        # 2. CatBoost（修复后：DataFrame输入+列名匹配cat_features）
        cat_model = cb.CatBoostClassifier(**CAT_PARAMS)
        cat_model.fit(X_train_cat, y_train, eval_set=[(X_val_cat, y_val)], verbose=0)
        base_preds[val_idx, 1] = cat_model.predict_proba(X_val_cat)[:, 1]
        
        # 3. LightGBM（量化+连续特征，修复：删除不支持的参数）
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        # 修复：仅保留文件要求的核心参数，删除 early_stopping_rounds 和 categorical_feature
        lgb_model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_val, y_val)], 
            eval_metric="binary_logloss"  # 可选：指定二分类评估指标（符合文件二分类任务）
        )
        base_preds[val_idx, 2] = lgb_model.predict_proba(X_val)[:, 1]
        
        # 4. MLP（仅连续特征，用numpy数组）
        mlp_features = ["PCI", "AADT", "YSM", "AR", "R", "IRI", "Com_Damage", "Rain_AADT"]
        X_train_mlp = df.iloc[train_idx][mlp_features].values
        X_val_mlp = df.iloc[val_idx][mlp_features].values
        mlp_model = MLPClassifier(**MLP_PARAMS)
        mlp_model.fit(X_train_mlp, y_train)
        base_preds[val_idx, 3] = mlp_model.predict_proba(X_val_mlp)[:, 1]
    
    # 第二层：逻辑回归元模型（第二问.pdf1-14节）
    lr_model = LogisticRegression(**LR_PARAMS)
    lr_model.fit(base_preds, y)
    y_prob = lr_model.predict_proba(base_preds)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # 输出预测结果（含论文要求的字段）
    result_df = df[[ID_COL, label_col]].copy()
    result_df["y_hat_prob"] = y_prob.round(4)
    result_df["y_hat_i"] = y_pred
    result_path = os.path.join(OUTPUT_PRED_PATH, "维护需求预测结果.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"✅ 预测结果保存至：{result_path}")
    
    # SHAP特征分析（B2025070938351.pdf5.1.4节要求）
    explainer = shap.TreeExplainer(xgb_model)  # 用XGBoost做特征重要性
    shap_values = explainer.shap_values(df[model_features])
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, df[model_features], feature_names=model_features, plot_type='bar')
    shap_path = os.path.join(OUTPUT_PRED_PATH, "特征重要性SHAP图.png")
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP分析图保存至：{shap_path}")
    
    return result_df, lr_model

# -------------------------- 5. 预测性能评估（第二问.pdf1-21节6个指标） --------------------------
def evaluate_prediction(y_true, y_pred, y_prob):
    metrics = {
        "准确率(Accuracy)": accuracy_score(y_true, y_pred),
        "精确率(Precision)": precision_score(y_true, y_pred),
        "召回率(Recall)": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_prob),
        "Kappa系数": cohen_kappa_score(y_true, y_pred)
    }
    metrics_df = pd.DataFrame(metrics.items(), columns=["指标名称", "数值"])
    metrics_df["数值"] = metrics_df["数值"].round(4)
    metrics_df["论文参考目标值"] = ["≥0.85", "≥0.80", "≥0.85", "≥0.82", "≥0.90", "≥0.75"]
    
    # 保存评估结果
    eval_path = os.path.join(OUTPUT_PRED_PATH, "预测性能评估.csv")
    metrics_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    
    # 打印结果
    print("\n" + "="*50)
    print("预测性能评估（第二问.pdf1-21节）")
    print("="*50)
    print(metrics_df.to_string(index=False))
    return metrics_df

# -------------------------- 6. 预测主流程（衔接论文初步预处理） --------------------------
if __name__ == "__main__":
    
    safe_redirect()

    # 1. 加载并校验初步预处理数据
    df_preprocessed = load_and_validate_preprocessed_data(INPUT_PREPROCESSED_PATH)
    
    # 2. 补充论文要求的预处理（特征衍生）
    df_final, model_features, cat_features, scaler = supplement_preprocessing(df_preprocessed)
    
    # 3. Stacking集成预测
    result_df, lr_model = stacking_prediction(df_final, model_features, cat_features, LABEL_COL)
    
    # 4. 评估性能
    y_true = result_df[LABEL_COL].values
    y_pred = result_df["y_hat_i"].values
    y_prob = result_df["y_hat_prob"].values
    metrics_df = evaluate_prediction(y_true, y_pred, y_prob)
    
    print(f"\n✅ 维护需求预测代码执行完成！输出路径：{OUTPUT_PRED_PATH}")