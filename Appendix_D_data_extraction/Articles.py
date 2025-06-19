import pandas as pd

# Step 1: 读取主文章数据
articles = pd.read_csv("Articles_78.csv", usecols=[
    'ID', 'Title', 'Abstract', 'Keywords', 'Authors', 'Year', 'Publication_Type'
])

# Step 2: 读取其他字段并按 ID 合并
files_and_columns = [
    ("Source_Name.csv", "Source_Name"),
    ("Platform_Type.csv", "Platform_Type"),
    ("Research_Area.csv", "Research_Area"),
    ("System_Architecture.csv", "System_Architecture"),
    ("Agent_Role.csv", "Agent_Role"),
    ("Behavior_Mechanism.csv", "Behavior_Mechanism"),
    ("Evaluation_Method.csv", "Evaluation_Method"),
    ("Experimental_Design.csv", "Experimental_Design"),
    ("Challenge_Type.csv", "Challenge_Type"),
    ("Research_Gap.csv", "Research_Gap"),
    ("Evaluation_Limitations.csv", "Evaluation_Limitations"),
]

for file_name, column_name in files_and_columns:
    df = pd.read_csv(file_name, usecols=['ID', column_name])
    articles = pd.merge(articles, df, on='ID', how='left')

# Step 3: 导出最终合并结果
articles.to_csv("Articles.csv", index=False)
print("✅ Articles.csv 已成功生成。")
