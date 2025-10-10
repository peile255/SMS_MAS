import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- 配置全局字体 ----------
plt.rcParams['font.family'] = 'Times New Roman'

# 创建输出文件夹
os.makedirs('../2_RQ1_2/figures', exist_ok=True)

# 读取数据（示例，替换为你的文件路径）
df = pd.read_csv('../2_RQ1_2/keywords.csv')

# 排序取前10
df_sorted = df.sort_values(by='Frequency', ascending=False)
top_10 = df_sorted.head(10)

# 保存top10数据
top_10.to_csv('figures/RQ1_2_top_10_keywords.csv', index=False)

# 创建画布（不指定 figsize，用默认）
fig, ax = plt.subplots()

# 取频率和关键词，反转顺序使得绘图时最高频在上面
freqs = top_10['Frequency'][::-1]
keywords = top_10['Keyword'][::-1]

# 颜色渐变，归一化
colors = plt.cm.Blues(freqs / max(freqs))


# 绘制水平条形图
bars = ax.barh(
    keywords,
    freqs,
    color=colors,
    edgecolor='black',
    linewidth=0.7,
    height=0.7
)

# 条形右侧添加频率标签
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + max(freqs) * 0.03,
        bar.get_y() + bar.get_height() / 2,
        f'{int(width)}',
        va='center',
        ha='left',
        fontsize=11
    )

# 设置轴标签和标题
ax.set_xlabel('Frequency', labelpad=8, fontsize=12)
ax.set_ylabel('Keywords', labelpad=8, fontsize=12)
ax.set_title('Top 10 Keywords by Frequency', pad=10, fontsize=14, weight='bold')

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加浅色网格线
ax.xaxis.grid(True, linestyle='--', alpha=0.3)

# 设置y轴刻度标签字号
ax.tick_params(axis='y', labelsize=14,)

# 动态计算左边距，避免关键词标签被裁剪
max_label_len = max(len(str(label)) for label in keywords)
left_margin = 0.15 + max_label_len * 0.015
left_margin = min(max(left_margin, 0.15), 0.4)  # 限制范围

# 调整子图边距
fig.subplots_adjust(left=left_margin, right=0.95, top=0.9, bottom=0.15)

# 自动紧凑布局，防止内容重叠
fig.tight_layout()

# 保存高分辨率PNG文件
plt.savefig(
    'figures/RQ1_2_top_10_keywords.png',
    dpi=600,
    bbox_inches='tight',
    transparent=False
)

plt.show()

print("图表已保存至：figures/RQ1_2_top_10_keywords.png")
print("数据已保存至：figures/RQ1_2_top_10_keywords.csv")
