import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

# ---------- 全局字体和样式配置 ----------
plt.rcParams['font.family'] = 'Times New Roman'

# 创建输出文件夹
os.makedirs("../1_RQ1/figures", exist_ok=True)

# 读取数据文件
df = pd.read_csv('../1_RQ1/Articles.csv')

# 数据清洗：去除 Source_Name 为空的行
df = df.dropna(subset=['Venue'])
df['Source_Name'] = df['Venue'].str.strip()

# 统计来源平台出现频次
source_counts = Counter(df['Venue'])

# 构造Series，取前10个来源平台及其频次
source_series = pd.Series(dict(source_counts.most_common(10)))

# 保存统计数据到CSV文件
source_df = source_series.reset_index()
source_df.columns = ['Venue', 'Count']
source_df.to_csv('figures/RQ1_Publication_Platforms.csv', index=False)

print("数据已保存至：figures/RQ1_Publication_Platforms.csv")

# 创建画布（不传 figsize，默认大小）
fig, ax = plt.subplots()

# 数据
counts = source_series.values
platforms = source_series.index

# 颜色渐变，基于频次归一化
colors = plt.cm.Blues(counts / max(counts))

# 绘制水平条形图，条形高度适中
bars = ax.barh(
    platforms,
    counts,
    color=colors,
    edgecolor='black',
    linewidth=0.7,
    height=0.7
)

# 在条形右侧添加频次标签
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + max(counts) * 0.03,
        bar.get_y() + bar.get_height() / 2,
        f'{int(width)}',
        va='center',
        ha='left',
        fontsize=11
    )

# 设置轴标签和标题
ax.set_xlabel('Number of Publications', labelpad=8, fontsize=12)
ax.set_ylabel('Source Platform', labelpad=8, fontsize=12)
ax.set_title('Top 10 Publication Platforms', pad=10, fontsize=14, weight='bold')

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加浅色网格线（仅x轴）
ax.xaxis.grid(True, linestyle='--', alpha=0.3)

# 反转Y轴顺序，使最大平台排最上
ax.invert_yaxis()

# 设置y轴刻度标签字号
ax.tick_params(axis='y', labelsize=14)

# 动态计算左边距，避免y轴标签被裁剪
max_label_len = max(len(str(label)) for label in platforms)
left_margin = 0.15 + max_label_len * 0.015
left_margin = min(max(left_margin, 0.15), 0.4)  # 限制范围

# 调整子图边距
fig.subplots_adjust(left=left_margin, right=0.95, top=0.9, bottom=0.15)

# 自动紧凑布局，防止内容重叠
fig.tight_layout()

# 保存高清图（600dpi）
plt.savefig(
    "figures/RQ1_Publication_Platforms.png",
    dpi=600,
    bbox_inches='tight',
    transparent=False
)

plt.show()

print("图表已保存至：figures/RQ1_Publication_Platforms.png")
