import pandas as pd

# 输入输出文件名
input_file = 'Articles_546.csv'
output_file = 'Articles_332.csv'

try:
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 删除 Title、Abstract 或 Keywords 中有缺失值的行
    cleaned_df = df.dropna(subset=['Title', 'Abstract', 'Keywords'])

    # 另存为新 CSV，字段加引号
    cleaned_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 相当于 csv.QUOTE_ALL

    print(f"✅ Successfully removed rows with empty Title/Abstract/Keywords. Saved to '{output_file}'.")

except FileNotFoundError:
    print(f"❌ Error: File '{input_file}' not found.")
except Exception as e:
    print(f"❗ An unexpected error occurred: {e}")
