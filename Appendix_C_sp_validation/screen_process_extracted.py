import pandas as pd

# 输入和输出文件名
input_file = 'screen_process_validation_sample.csv'
output_file = 'screen_process_extracted.csv'

# 要提取的列
columns_to_extract = [
    'Article_Number',
    'Title',
    'ChatGPT',
    'DeepSeek',
    'Sum',
    'Related_Label',
    'Manual_Label',
    'AI_Prediction'
]

def extract_columns(input_path, output_path, columns):
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 提取指定列
    extracted_df = df[columns]

    # 保存到新CSV文件
    extracted_df.to_csv(output_path, index=False)
    print(f"提取完成，新文件已保存为: {output_path}")

if __name__ == '__main__':
    extract_columns(input_file, output_file, columns_to_extract)
