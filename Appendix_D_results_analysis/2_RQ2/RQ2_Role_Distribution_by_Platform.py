import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

# ---------- 全局字体和样式配置 ----------
plt.rcParams['font.family'] = 'Times New Roman'

# 创建输出文件夹
os.makedirs("figures", exist_ok=True)

# ---------- 读取数据 ----------
df = pd.read_csv("Articles.csv")

# ---------- 安全解析函数 ----------
def safe_parse(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except:
        return []

df['Agent_Role'] = df['Agent_Role'].apply(safe_parse)
df['Platform_Type'] = df['Platform_Type'].apply(safe_parse)

# ---------- 展开多标签字段 ----------
df_exp = df.explode('Platform_Type').explode('Agent_Role')

# ---------- 清洗空值 ----------
df_exp['Agent_Role'] = df_exp['Agent_Role'].astype(str).str.strip()
df_exp['Platform_Type'] = df_exp['Platform_Type'].astype(str).str.strip()
df_exp = df_exp[
    (df_exp['Agent_Role'].str.lower() != 'nan') &
    (df_exp['Agent_Role'].str.lower() != 'none') &
    (df_exp['Agent_Role'] != '') &
    (df_exp['Platform_Type'].str.lower() != 'nan') &
    (df_exp['Platform_Type'].str.lower() != 'none') &
    (df_exp['Platform_Type'] != '')
]

# ---------- 构建交叉表 ----------
role_platform_matrix = df_exp.groupby(['Platform_Type', 'Agent_Role']).size().unstack(fill_value=0)

# ---------- 保存数据 ----------
role_platform_matrix.to_csv('figures/RQ2_Role_Distribution_by_Platform.csv')
print("✅ 数据已保存至 figures/RQ2_Role_Distribution_by_Platform.csv")

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    role_platform_matrix,
    annot=True,
    fmt='g',
    cmap='Blues',
    linewidths=0.5,
    cbar_kws={'label': 'Frequency'},
    annot_kws={"color": "black"},
    ax=ax
)

# ---------- 图表样式 ----------
ax.set_title('Heatmap of MAS Roles by Platform Type', fontsize=14, weight='bold', pad=12)
ax.set_xlabel('Agent Role', labelpad=8, fontsize=12)
ax.set_ylabel('Platform Type', labelpad=8, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

fig.tight_layout()

# ---------- 保存图像 ----------
plt.savefig(
    'figures/RQ2_Role_Distribution_by_Platform.png',
    dpi=600,
    bbox_inches='tight',
    transparent=False
)

plt.show()
print("✅ 图表已保存至 figures/RQ2_Role_Distribution_by_Platform.png")
