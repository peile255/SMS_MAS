import pandas as pd

# 输入文件
file_positive = 'Articles_86_labeled.csv'    # AI筛选保留的
file_negative = 'Articles_246_labeled.csv'   # AI筛选剔除的
output_file = 'screen_process_validation_sample.csv'

# 读取数据
df_pos = pd.read_csv(file_positive)
df_neg = pd.read_csv(file_negative)

# 随机抽取样本
sample_pos = df_pos.sample(n=9, random_state=42)   # 固定种子以确保可复现
sample_neg = df_neg.sample(n=24, random_state=42)

# 添加标签用于追踪来源
sample_pos['AI_Prediction'] = 'Relevant'
sample_neg['AI_Prediction'] = 'Irrelevant'

# 合并并打乱顺序
df_sample = pd.concat([sample_pos, sample_neg], ignore_index=True)
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# 输出为新 CSV 文件
df_sample.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Validation sample saved to '{output_file}'")
