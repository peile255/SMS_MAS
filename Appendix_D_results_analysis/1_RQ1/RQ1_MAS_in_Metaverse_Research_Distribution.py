import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- 全局字体和样式配置 ----------
plt.rcParams['font.family'] = 'Times New Roman'

# 创建输出文件夹
os.makedirs("../1_RQ1/figures", exist_ok=True)

# 读取关键词数据
df = pd.read_csv('keywords.csv')  # 你的文件名是 keyword.csv

# 定义元宇宙平台类别及其关键词（关键词基于你的数据和常见相关词调整）
metaverse_categories = {
    'Game Engine': ['rendering', 'real time', 'unity', 'game'],
    'Social VR': ['social', 'avatar', 'human computer', 'vrchat'],
    'Blockchain': ['blockchain', 'nft', 'digital economy'],
    'Industrial': ['uav', 'vehicle', 'industrial', 'manufacturing', 'robot'],
    'Infrastructure': ['wireless', 'edge', 'cloud', 'computing', 'iot', 'network'],
}

# 汇总每个类别关键词频次
category_counts = {}
for category, keywords in metaverse_categories.items():
    # 构造正则匹配串，关键词间用|分隔，忽略大小写
    pattern = '|'.join(keywords)
    mask = df['Keyword'].str.contains(pattern, case=False, na=False)
    category_counts[category] = df.loc[mask, 'Frequency'].sum()

# 转换为DataFrame，计算百分比并排序
count_df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['Count'])
count_df = count_df.sort_values('Count', ascending=False)
count_df['Percentage'] = count_df['Count'] / count_df['Count'].sum() * 100

# 保存至CSV文件
count_df.to_csv('figures/RQ1_MAS_in_Metaverse_Research_Distribution.csv')
print("数据已保存至：figures/RQ1_MAS_in_Metaverse_Research_Distribution.csv")

# 设置图像尺寸（IEEE单栏宽度）
fig, ax = plt.subplots(figsize=(3.35, 3.6), dpi=600)

# 饼图颜色（大值颜色更深）
colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(count_df)))  # 深蓝 → 浅蓝
explode = [0.03] * len(count_df)

# 绘制饼图（无标签）
wedges, _ = ax.pie(
    count_df['Count'],
    labels=None,
    colors=colors,
    explode=explode,
    startangle=110,
    wedgeprops=dict(edgecolor='white', linewidth=1.2, alpha=0.95)
)

# 图例：类别名 + 数量 + 百分比
legend_labels = [
    f"{label} ({count}, {pct:.1f}%)"
    for label, count, pct in zip(count_df.index, count_df['Count'], count_df['Percentage'])
]

ax.legend(
    wedges,
    legend_labels,
    title='Platform Categories',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=True,
    framealpha=0.85,
    edgecolor='gray'
)

# 添加标题
ax.set_title(
    'MAS in the Metaverse Research Distribution',
    pad=10,
    fontsize=10,
    weight='bold'
)

# 紧凑布局
fig.tight_layout()
plt.subplots_adjust(bottom=0.25)

# 保存图像
plt.savefig(
    'figures/RQ1_MAS_in_Metaverse_Research_Distribution.png',
    dpi=600,
    bbox_inches='tight',
    transparent=False
)

plt.show()
print("图表已保存至：figures/RQ1_MAS_in_Metaverse_Research_Distribution.png")