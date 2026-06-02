import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 准备数据
data = {
    '类别': ['0', '1', 'macro avg', 'weighted avg'],
    'precision': [1.00, 1.00, 1.00, 1.00],
    'recall': [1.00, 1.00, 1.00, 1.00],
    'f1-score': [1.00, 1.00, 1.00, 1.00]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置图形风格
plt.style.use('seaborn-v0_8-talk')

# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱状图位置
bar_width = 0.25
index = np.arange(len(df['类别']))

# 绘制柱状图
rects1 = ax.bar(index - bar_width, df['precision'], bar_width, label='Precision', color='#4CAF50', edgecolor='black')
rects2 = ax.bar(index, df['recall'], bar_width, label='Recall', color='#2196F3', edgecolor='black')
rects3 = ax.bar(index + bar_width, df['f1-score'], bar_width, label='F1-Score', color='#FFC107', edgecolor='black')

# 添加标题和标签
ax.set_title('分类模型性能指标', fontsize=16, pad=20)
ax.set_xlabel('类别', fontsize=14, labelpad=10)
ax.set_ylabel('分数', fontsize=14, labelpad=10)
ax.set_xticks(index)
ax.set_xticklabels(df['类别'], fontsize=12)
ax.legend(fontsize=12)

# 设置y轴范围
ax.set_ylim(0.99, 1.005)

# 在柱状图上添加数值标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# 添加准确率信息
accuracy_text = f'准确率 (Accuracy): 1.00'
plt.figtext(0.5, 0.01, accuracy_text, ha='center', fontsize=12, bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存图形
# plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')
