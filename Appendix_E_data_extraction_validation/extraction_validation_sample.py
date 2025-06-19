import pandas as pd
import random
import os

# 确保结果目录存在
os.makedirs("results", exist_ok=True)

# 读取 Articles.csv（确保文件路径正确）
df = pd.read_csv("Articles.csv")

# 定义需要验证的字段
fields_to_validate = [
    "Platform_Type", "Research_Area", "System_Architecture", "Agent_Role",
    "Behavior_Mechanism", "Evaluation_Method", "Experimental_Design",
    "Challenge_Type", "Research_Gap", "Evaluation_Limitations"
]

# 随机选取 8 篇文章
sampled_articles = df.sample(n=8, random_state=42)

# 创建用于人工验证的表格（长表格式）
records = []

for _, row in sampled_articles.iterrows():
    for field in fields_to_validate:
        records.append({
            "ID": row["ID"],
            "Field": field,
            "AI_Extraction": row[field],
            "Human_Annotation": "",
            "Resolution": "",
            "Comments": ""
        })

# 转换为 DataFrame
validation_df = pd.DataFrame(records)

# 保存为 CSV 文件
validation_df.to_csv("extraction_validation_sample.csv", index=False)

print("✅ extraction_validation_sample.csv 已保存到 目录。")
