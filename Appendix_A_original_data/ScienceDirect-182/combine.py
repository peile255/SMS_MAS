import pandas as pd
import os

# 定义要合并的文件名列表
file_names = ['bib_output.csv', 'bib2_output.csv']

# 读取并合并所有 CSV 文件
dataframes = []
for file in file_names:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dataframes.append(df)
    else:
        print(f"Warning: {file} not found and will be skipped.")

# 合并为一个 DataFrame
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv('final_sd.csv', index=False)
    print("Successfully combined into final_sd.csv")
else:
    print("No files were combined. Please check file names.")
