import pandas as pd

# 读取CSV
df = pd.read_csv("titles1_clean.csv")

# 去掉 abstract 或 keywords 为空的行（包括空字符串或缺失值）
df_cleaned = df.dropna(subset=['abstract', 'keywords'])  # 去除NaN
df_cleaned = df_cleaned[(df_cleaned['abstract'].str.strip() != '') & (df_cleaned['keywords'].str.strip() != '')]

# 保存为新CSV文件
df_cleaned.to_csv("final_acm.csv", index=False)

print("✅ 已生成过滤后的文件 'final_acm.csv'")
