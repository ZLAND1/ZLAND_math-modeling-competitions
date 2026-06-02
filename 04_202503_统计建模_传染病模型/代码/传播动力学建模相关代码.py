import pymc3 as pmimport theano.tensor as ttimport matplotlib.pyplot as plt
# 改进SEIR模型的构建与参数估计# 假设population_data为包含各地区人口数量的数据集# infection_data为包含各地区感染人数、时间等信息的数据集
population_data = pd.read_csv('population_data.csv')
infection_data = pd.read_csv('infection_data.csv')
# 定义改进SEIR模型def improved_seir_model(population, infection):
    with pm.Model() as model:
        # 参数先验分布
        beta = pm.TruncatedNormal('beta', mu=0.3, sd=0.1, lower=0)
        sigma = pm.Gamma('sigma', alpha=2, beta=1)  # 潜伏期倒数

        # 状态变量
        S = pm.Normal('S', mu=population, sd=100)
        E = pm.Normal('E', mu=0, sd=100)
        I = pm.Normal('I', mu=infection.iloc[0]['感染人数'], sd=100)
        R = pm.Normal('R', mu=0, sd=100)

        # 空间传播项参数
        P = population_data['人口数'].values
        d = np.random.rand(len(P), len(P))  # 假设距离矩阵随机生成，实际需根据地理信息计算
        alpha = 1.5
        K_ij = np.array([[P[i] * P[j] / (d[i][j] ** alpha) for j in range(len(P))] for i in range(len(P))])

        # 模型方程
        def update_seir(t):
            S_t = S[t - 1] - beta * (I[t - 1] / population) * S[t - 1] + tt.sum(K_ij[:, t - 1] * (S[:, t - 1] - S[t - 1])) + 0.01 * population - 0.01 * S[t - 1]
            E_t = beta * (I[t - 1] / population) * S[t - 1] - sigma * E[t - 1]
            I_t = sigma * E[t - 1] - 0.1 * I[t - 1]
            R_t = 0.1 * I[t - 1] + R[t - 1]
            return S_t, E_t, I_t, R_t

        S_ = pm.Deterministic('S_', tt.stack([S[0]] + [update_seir(t)[0] for t in range(1, len(infection))]))
        E_ = pm.Deterministic('E_', tt.stack([E[0]] + [update_seir(t)[1] for t in range(1, len(infection))]))
        I_ = pm.Deterministic('I_', tt.stack([I[0]] + [update_seir(t)[2] for t in range(1, len(infection))]))
        R_ = pm.Deterministic('R_', tt.stack([R[0]] + [update_seir(t)[3] for t in range(1, len(infection))]))

        # 似然函数
        likelihood = pm.Normal('likelihood', mu=I_, sd=1, observed=infection['感染人数'].values)

        # 采样
        trace = pm.sample(2000, tune=1000, chains=2, cores=2)
    return model, trace

model, trace = improved_seir_model(population_data['人口数'].values, infection_data)
# 随机森林传播风险预测模型from sklearn.ensemble import RandomForestClassifierfrom sklearn.model_selection import GridSearchCVfrom sklearn.inspection import permutation_importanceimport shap
# 假设rf_data为包含所有特征和目标变量（传播风险）的数据集
rf_data = pd.read_csv('rf_data.csv')

X_rf = rf_data.drop('传播风险', axis=1)
y_rf = rf_data['传播风险']
# 特征工程# 假设已经完成特征选择和处理
# 随机森林模型训练
rf = RandomForestClassifier()
rf.fit(X_rf, y_rf)
# 特征重要性排序
importances = permutation_importance(rf, X_rf, y_rf)
sorted_indices = np.argsort(importances.importances_mean)[::-1]print("特征重要性排序：")for f in sorted_indices:
    print(f"{X_rf.columns[f]}: {importances.importances_mean[f]}")
# 超参数调优
param_grid = {
    'n_estimators': [50, 100, 200],
   'max_depth': [None, 10, 20, 30],
   'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_rf, y_rf)print("最优参数：", grid_search.best_params_)print("最优得分：", grid_search.best_score_)
# 可解释性增强（SHAP值分析）
explainer = shap.Explainer(rf)
shap_values = explainer(X_rf)
shap.summary_plot(shap_values, X_rf)