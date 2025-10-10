import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast
import os

# ---------- 配置全局字体 ----------
plt.rcParams['font.family'] = 'Times New Roman'

# 1. 创建 figures 文件夹
os.makedirs('figures', exist_ok=True)

# 2. 读取数据
df = pd.read_csv('Articles.csv')

# 3. 清洗：去除 Agent_Role 为空的行
df = df.dropna(subset=['Application_Domain'])

# 4. 解析 Agent_Role（字符串转列表）
def try_parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []

df['Application_Domain'] = df['Application_Domain'].apply(try_parse)

# 5. 拆分多标签为一维列表，统计频次
roles_flat = [role.strip() for sublist in df['Application_Domain'] for role in sublist if str(role).strip().lower() not in ['nan', 'none', '']]
role_counts = Counter(roles_flat)

# 6. 构造 Series 并降序排列
role_series = pd.Series(role_counts).sort_values(ascending=False)

# 7. 保存统计数据为 CSV
role_df = role_series.reset_index()
role_df.columns = ['Application_Domain', 'Count']
role_df.to_csv('figures/RQ3_Application_Domain.csv', index=False)
print("数据已保存至：figures/RQ3_Application_Domain.csv")

# ---------- 图表绘制部分 ----------

# 创建画布
fig, ax = plt.subplots()

# 数据
counts = role_series.values
roles = role_series.index

# 渐变色处理
colors = plt.cm.Blues(counts / max(counts))

# 绘制水平条形图
bars = ax.barh(
    roles,
    counts,
    color=colors,
    edgecolor='black',
    linewidth=0.7,
    height=0.7
)

# 在条形右侧标注频次
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

# 设置标签与标题
ax.set_xlabel('Frequency', labelpad=8, fontsize=12)
ax.set_ylabel('Application Domain', labelpad=8, fontsize=12)
ax.set_title('Application Domain Distribution', pad=10, fontsize=14, weight='bold')

# 去除顶部和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加浅色网格线
ax.xaxis.grid(True, linestyle='--', alpha=0.3)

# 角色从高到底排序
ax.invert_yaxis()

# Y轴刻度标签字号
ax.tick_params(axis='y', labelsize=14)

# 自动计算左边距避免裁剪
max_label_len = max(len(str(label)) for label in roles)
left_margin = 0.15 + max_label_len * 0.015
left_margin = min(max(left_margin, 0.15), 0.4)

# 调整边距
fig.subplots_adjust(left=left_margin, right=0.95, top=0.9, bottom=0.15)

# 保存高质量图像
plt.savefig(
    'figures/RQ3_Application_Domain.png',
    dpi=600,
    bbox_inches='tight',
    transparent=False
)

plt.show()
print("图表已保存至：figures/RQ3_Application_Domain.png")
