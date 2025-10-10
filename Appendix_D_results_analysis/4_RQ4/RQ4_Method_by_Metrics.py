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

df['Evaluation_Method'] = df['Evaluation_Method'].apply(safe_parse)
df['Metrics'] = df['Metrics'].apply(safe_parse)

# ---------- 展开多标签字段 ----------
df_exp = df.explode('Evaluation_Method').explode('Metrics')

# ---------- 清洗空值 ----------
df_exp['Evaluation_Method'] = df_exp['Evaluation_Method'].astype(str).str.strip()
df_exp['Metrics'] = df_exp['Metrics'].astype(str).str.strip()
df_exp = df_exp[
    (df_exp['Evaluation_Method'].str.lower() != 'nan') &
    (df_exp['Evaluation_Method'].str.lower() != 'none') &
    (df_exp['Evaluation_Method'] != '') &
    (df_exp['Metrics'].str.lower() != 'nan') &
    (df_exp['Metrics'].str.lower() != 'none') &
    (df_exp['Metrics'] != '')
]

# ---------- 构建交叉表 ----------
method_metric_matrix = df_exp.groupby(['Evaluation_Method', 'Metrics']).size().unstack(fill_value=0)

# ---------- 保存数据 ----------
method_metric_matrix.to_csv('figures/RQ4_Method_by_Metrics.csv')
print("✅ 数据已保存至 figures/RQ4_Method_by_Metrics.csv")

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    method_metric_matrix,
    annot=True,
    fmt='g',
    cmap='Purples',
    linewidths=0.5,
    cbar_kws={'label': 'Frequency'},
    annot_kws={"color": "black"},
    ax=ax
)

# ---------- 图表样式 ----------
ax.set_title('Heatmap of Evaluation Methods by Metrics', fontsize=14, weight='bold', pad=12)
ax.set_xlabel('Metrics', labelpad=8, fontsize=12)
ax.set_ylabel('Evaluation Method', labelpad=8, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

fig.tight_layout()

# ---------- 保存图像 ----------
plt.savefig(
    'figures/RQ4_Method_by_Metrics.png',
    dpi=600,
    bbox_inches='tight',
    transparent=False
)

plt.show()
print("✅ 图表已保存至 figures/RQ4_Method_by_Metrics.png")
