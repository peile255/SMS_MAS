import pandas as pd

# 读取原始 CSV 文件
input_file = 'Articles_246_filtered_sum.csv'
output_file = 'Articles_246_labeled.csv'

# 加载数据
df = pd.read_csv(input_file)

# 添加 Related_Label 和 Manual_Label 列
df['Related_Label'] = 'No'     # 默认全为 Yes
df['Manual_Label'] = ''         # 留空等待人工标注

# 保存新文件
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Labeled file saved as '{output_file}'")
