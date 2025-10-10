import pandas as pd
import ast
import os

# ---------- 创建输出文件夹 ----------
os.makedirs("figures", exist_ok=True)

# ---------- 读取数据 ----------
df = pd.read_csv("Articles.csv")

# ---------- 安全解析多标签字段 ----------
def safe_parse(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except:
        return []

df['Agent_Role'] = df['Agent_Role'].apply(safe_parse)
df['Platform_Type'] = df['Platform_Type'].apply(safe_parse)

# ---------- 展开多标签 ----------
df_exp = df.explode('Agent_Role').explode('Platform_Type')

# ---------- 清洗空白和非法值 ----------
df_exp['Agent_Role'] = df_exp['Agent_Role'].astype(str).str.strip()
df_exp['Platform_Type'] = df_exp['Platform_Type'].astype(str).str.strip()
df_exp = df_exp[
    (df_exp['Agent_Role'].str.lower() != 'nan') &
    (df_exp['Platform_Type'].str.lower() != 'nan') &
    (df_exp['Agent_Role'] != '') & (df_exp['Platform_Type'] != '')
]

# ---------- 📄 statistics_1.csv: 每个 Agent_Role -> 文章 ID 列表 ----------
stats1 = df_exp.groupby('Agent_Role')['ID'].apply(lambda x: sorted(x.unique().tolist())).reset_index()
stats1.columns = ['Agent_Role', 'Article_IDs']
stats1.to_csv('figures/statistics_1.csv', index=False)

# ---------- 📄 statistics_2.csv: 每个 (Platform_Type, Agent_Role) -> 文章 ID 列表 ----------
stats2 = df_exp.groupby(['Platform_Type', 'Agent_Role'])['ID'].apply(lambda x: sorted(x.unique().tolist())).reset_index()
stats2.columns = ['Platform_Type', 'Agent_Role', 'Article_IDs']
stats2.to_csv('figures/statistics_2.csv', index=False)

print("✅ statistics_1.csv 和 statistics_2.csv 已生成并保存在 figures 文件夹中。")
