import tensorflow as tffrom tensorflow.keras.models import Sequentialfrom tensorflow.keras.layers import LSTM, Dense, Dropoutimport numpy as npimport matplotlib.pyplot as plt
# LSTM时间序列预测模型# 假设lstm_data为包含发病数、迁入量、温度等特征的时间序列数据集
lstm_data = pd.read_csv('lstm_data.csv')
# 数据准备
time_steps = 30
features = 12
data = lstm_data.values
X = []
y = []for i in range(len(data) - time_steps - 7):
    X.append(data[i:i + time_steps, :features])
    y.append(data[i + time_steps:i + time_steps + 7, 0])  # 假设发病数在第一列
X = np.array(X)
y = np.array(y)
# 划分训练集和验证集
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
# 构建LSTM模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, features)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(7))
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
# 训练模型
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
# 绘制训练曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# 周期性特征提取from scipy.fftpack import fft
# 假设取历史发病数序列
historical_cases = lstm_data['发病数'].values
n = len(historical_cases)
yf = fft(historical_cases)
xf = np.linspace(0.0, 1.0 / (2.0 * (n / len(historical_cases))), n // 2)
# 识别显著周期# 这里简单根据频谱峰值判断，实际需更复杂处理
peak_indices = np.argpartition(np.abs(yf[0:n // 2]), -5)[-5:]
periods = 1 / xf[peak_indices]print("显著周期：", periods)
# 防控策略的量化分析相关代码示例# 传播阻断措施的成本 - 效果比计算# 假设已知各措施的实施成本、发病率降低幅度和人口数
implementation_costs = [80, 150, 200]  # 万元/区县
incidence_reduction_rates = [0.25, 0.35, 0.40]
population = 1000000  # 百万人口区县

cost_effectiveness_ratios = []for cost, reduction in zip(implementation_costs, incidence_reduction_rates):
    cost_effectiveness = cost / (0.01 * reduction * population)  # 假设原发病率为1%
    cost_effectiveness_ratios.append(cost_effectiveness)print("成本效果比：", cost_effectiveness_ratios)
# 国际输入风险动态评估模型# 假设已知来源国当前疫情强度、变异株传播优势因子、直飞航班频率和地理距离
I = [0.5, 0.8, 0.3]  # 来源国当前疫情强度（病例数/百万人口，标准化后）
V = [1.1, 1.3, 1.0]  # 变异株传播优势因子
C = [3, 5, 2]  # 直飞航班频率（每周班次）
D = [0.2, 0.4, 0.1]  # 地理距离（按机场直线距离，标准化为0-1）

risk_scores = []for i in range(len(I)):
    risk = 0.4 * I[i] + 0.3 * V[i] + 0.2 * C[i] + 0.1 * D[i]
    risk_scores.append(risk)print("风险评分：", risk_scores)