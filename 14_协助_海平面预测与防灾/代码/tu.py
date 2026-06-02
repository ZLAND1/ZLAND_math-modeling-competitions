
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# -------------------------- 图1：海平面预测与入侵概率热力图 --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 1.1 海平面预测曲线
years_hist = np.arange(1990, 2025)
sea_level_hist = 0.02*(years_hist-1990) + 0.05*np.sin((years_hist-1990)*2*np.pi/12) + np.random.normal(0, 0.01, len(years_hist))
years_pred = np.arange(2025, 2076)
sea_level_pred = 0.025*(years_pred-1990) + 0.06*np.sin((years_pred-1990)*2*np.pi/12)

ax1.plot(years_hist, sea_level_hist, 'b-', label='历史数据', alpha=0.7)
ax1.plot(years_pred, sea_level_pred, 'r--', label='预测趋势', linewidth=2)
ax1.axhline(sea_level_pred[-1], color='gray', linestyle=':', label=f'2075年预测：{sea_level_pred[-1]:.2f}m')
ax1.set_xlabel('年份', fontsize=11)
ax1.set_ylabel('海平面高度（相对于1990年，m）', fontsize=11)
ax1.set_title('1990-2075年沿海城市海平面变化预测', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)

# 1.2 入侵概率热力图（10个沿海城市×2025-2030年）
cities = [f'城市{i}' for i in range(1, 11)]
years_prob = np.arange(2025, 2031)
prob_matrix = np.random.uniform(0.1, 0.8, size=(len(cities), len(years_prob)))  # 模拟概率矩阵
prob_df = pd.DataFrame(prob_matrix, index=cities, columns=years_prob)

sns.heatmap(prob_df, ax=ax2, cmap='YlOrRd', annot=True, fmt='.2f', 
            cbar_kws={'label': '海水入侵概率'}, linewidths=0.5)
ax2.set_title('2025-2030年沿海城市海水入侵概率预测', fontsize=13)
ax2.set_xlabel('年份', fontsize=11)
ax2.set_ylabel('城市', fontsize=11)

plt.tight_layout()
plt.savefig('海平面预测与入侵概率热力图.png')
plt.show()



# -------------------------- 图2：短长期损失评估对比图 --------------------------
plt.figure(figsize=(12, 6))

# 模拟10个城市的短期（10年）和长期（50年）经济损失（单位：亿元）
cities = [f'城市{i}' for i in range(1, 11)]
short_loss = np.random.uniform(5, 30, size=10)  # 短期损失
long_loss = short_loss * np.random.uniform(3, 8, size=10)  # 长期损失（短期的3-8倍）

x = np.arange(len(cities))
width = 0.35

plt.bar(x - width/2, short_loss, width, label='短期损失（2025-2035）', color='#4CAF50')
plt.bar(x + width/2, long_loss, width, label='长期损失（2025-2075）', color='#FF9800')

plt.xlabel('沿海城市', fontsize=11)
plt.ylabel('经济损失（亿元）', fontsize=11)
plt.title('沿海城市海水入侵短长期经济损失对比', fontsize=13)
plt.xticks(x, cities, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('短长期损失评估对比图.png')
plt.show()
#plt.savefig('短长期损失评估对比图.png')


# -------------------------- 图3：综合风险分级热力图 --------------------------
plt.figure(figsize=(10, 8))

# 模拟15个城市的综合风险得分（0-10，越高风险越大）
cities = [f'城市{i}' for i in range(1, 16)]
risk_factors = ['海平面上升', '地面沉降', '地下水位', '基础设施脆弱性', '人口密度']
risk_matrix = np.random.randint(1, 11, size=(len(cities), len(risk_factors)))  # 风险矩阵

# 自定义风险颜色映射（低-中-高）
cmap = LinearSegmentedColormap.from_list('risk_cmap', ['#43A047', '#FFB300', '#E53935'])

sns.heatmap(risk_matrix, annot=True, cmap=cmap, fmt='d', 
            xticklabels=risk_factors, yticklabels=cities,
            cbar_kws={'label': '风险得分（0-10）'}, linewidths=0.8)
plt.title('沿海城市海水入侵综合风险分级评估', fontsize=13)
plt.xlabel('风险因子', fontsize=11)
plt.ylabel('城市', fontsize=11)
plt.tight_layout()
plt.savefig('综合风险分级热力图.png')
plt.show()



# -------------------------- 图4：防波堤建设规划成本-效益分析 --------------------------
plt.figure(figsize=(12, 6))

# 模拟防波堤高度与成本、风险降低率的关系
height = np.arange(1, 6.1, 0.5)  # 防波堤高度（m）
cost = 50 + 30*height + 5*height**2  # 成本（千万元）
risk_reduction = 10 + 25*height - 2*height**2  # 风险降低率（%），非线性增长

ax1 = plt.gca()
ax1.plot(height, cost, 'b-', marker='o', label='建设成本')
ax1.set_xlabel('防波堤高度（m）', fontsize=11)
ax1.set_ylabel('建设成本（千万元）', fontsize=11, color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(height, risk_reduction, 'r--', marker='s', label='风险降低率')
ax2.set_ylabel('风险降低率（%）', fontsize=11, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(0, 100)

plt.title('防波堤高度与成本-风险降低率关系', fontsize=13)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='center right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('防波堤建设规划成本-效益分析.png')
plt.show()




# -------------------------- 图5：功能区搬迁与防灾优化效果对比 --------------------------
plt.figure(figsize=(10, 6))

# 模拟5类功能区优化前后的风险值（0-10）
zones = ['居民区', '工业区', '商业区', '农业区', '生态保护区']
before_risk = np.random.uniform(6, 9, size=5)  # 优化前风险
after_risk = before_risk * np.random.uniform(0.3, 0.6, size=5)  # 优化后风险（降低40%-70%）

x = np.arange(len(zones))
width = 0.4

plt.bar(x - width/2, before_risk, width, label='优化前风险', color='#F44336', alpha=0.8)
plt.bar(x + width/2, after_risk, width, label='优化后风险', color='#2196F3', alpha=0.8)

plt.axhline(np.mean(before_risk), color='#F44336', linestyle='--', label=f'平均初始风险：{np.mean(before_risk):.2f}')
plt.axhline(np.mean(after_risk), color='#2196F3', linestyle='--', label=f'平均优化后风险：{np.mean(after_risk):.2f}')

plt.xlabel('城市功能区', fontsize=11)
plt.ylabel('风险值（0-10）', fontsize=11)
plt.title('功能区搬迁与防灾优化前后风险对比', fontsize=13)
plt.xticks(x, zones)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('功能区搬迁与防灾优化效果对比.png')
plt.show()
