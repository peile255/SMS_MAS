import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv("keywords_wordcloud.csv")

# 2. 保存为 Excel 文件（xlsx 格式）
df.to_excel("keywords_wordcloud.xlsx", index=False)
