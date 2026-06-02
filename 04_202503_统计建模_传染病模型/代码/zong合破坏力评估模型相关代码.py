from sklearn.decomposition import PCAimport numpy as np
# 假设impact_data为包含8个破坏力指标的数据集
impact_data = pd.read_csv('impact_data.csv')
# 数据标准化
scaler = StandardScaler()
standardized_impact = scaler.fit_transform(impact_data)
# 主成分分析
pca = PCA()
pca.fit(standardized_impact)
# 计算累计方差贡献率
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
# 选择主成分
n_components = np.argwhere(cumulative_variance > 0.85)[0][0] + 1  # 选择累计方差贡献超85%的主成分
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(standardized_impact)
# 计算破坏力指数
eigenvalues = pca.explained_variance_
weights = eigenvalues / np.sum(eigenvalues)

PC1 = principal_components[:, 0]
PC2 = principal_components[:, 1]
PC3 = principal_components[:, 2]

DI = weights[0] * PC1 + weights[1] * PC2 + weights[2] * PC3