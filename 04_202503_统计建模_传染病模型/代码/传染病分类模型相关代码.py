from sklearn.tree import DecisionTreeClassifierfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import accuracy_score
# 生物学分类的双层决策树模型# 假设biological_data为包含病原体类型、基因组稳定性等特征的数据集# '病原体类型'字段包含病毒、细菌、真菌、寄生虫等# '基因组稳定性'字段区分RNA病毒和DNA病毒等# '耐药性指数'等其他字段类似
biological_data = pd.read_csv('biological_data.csv')
# 一级分类：病原体类型
clf1 = DecisionTreeClassifier()
X1 = biological_data[['病原体类型']]
y1 = biological_data['病原体类型']
clf1.fit(X1, y1)
# 二级分类：对病毒和细菌进一步细分
virus_data = biological_data[biological_data['病原体类型'] == '病毒']
bacteria_data = biological_data[biological_data['病原体类型'] == '细菌']
# 对病毒分类
X_virus = virus_data[['基因组稳定性']]
y_virus = virus_data['病毒类型（RNA/DNA）']
clf_virus = DecisionTreeClassifier()
clf_virus.fit(X_virus, y_virus)
# 对细菌分类
X_bacteria = bacteria_data[['耐药性指数']]
y_bacteria = bacteria_data['细菌类型（革兰氏阳性/阴性）']
clf_bacteria = DecisionTreeClassifier()
clf_bacteria.fit(X_bacteria, y_bacteria)
# 统计特征分类的逻辑回归模型# 假设statistical_data为包含传播途径、病原体直径等特征的数据集
statistical_data = pd.read_csv('statistical_data.csv')

X = statistical_data[['病原体直径', '生存环境', '宿主接触频率',  # 其他特征
                      ]]
y = statistical_data['传播途径']

logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
accuracy = accuracy_score(y, y_pred)print("逻辑回归模型分类准确率：", accuracy)