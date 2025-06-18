import pandas as pd

# 输入输出文件名
input_file = 'Articles_558.csv'
output_file = 'Articles_546.csv'

try:
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 去掉 Title 重复的行，保留第一次出现的
    filtered_df = df.drop_duplicates(subset='Title', keep='first')

    # 保存为新的 CSV 文件
    filtered_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 即 csv.QUOTE_ALL

    print(f"✅ Successfully removed duplicate titles. Saved to '{output_file}'.")

except FileNotFoundError:
    print(f"❌ Error: File '{input_file}' not found.")
except Exception as e:
    print(f"❗ An unexpected error occurred: {e}")
