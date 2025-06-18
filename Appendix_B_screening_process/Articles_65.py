import pandas as pd

# 读取原始CSV文件
df = pd.read_csv("Articles_86.csv")

# 筛选 Year >= 2015 的行
filtered_df = df[df["Year"] >= 2015]

# 将筛选结果保存为新的CSV文件
filtered_df.to_csv("Articles_65.csv", index=False)

print("筛选完成，保存为 'Articles_65.csv'")
