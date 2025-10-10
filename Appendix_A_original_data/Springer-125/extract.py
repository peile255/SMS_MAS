import pandas as pd
import csv  # 用于设置 quoting 规则

# 文件路径
input_file = 'SearchResults.csv'
output_file = 'final_springer.csv'

# 要提取的列
columns_to_extract = ['Title']

try:
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 检查列是否存在
    missing_cols = [col for col in columns_to_extract if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing: {missing_cols}")
    else:
        # 提取指定列
        extracted_df = df[columns_to_extract]

        # 保存为新 CSV，所有字段用双引号包裹
        extracted_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Successfully extracted columns to '{output_file}' with proper quoting.")
except FileNotFoundError:
    print(f"❌ Error: File '{input_file}' not found.")
except Exception as e:
    print(f"❗ An unexpected error occurred: {e}")
