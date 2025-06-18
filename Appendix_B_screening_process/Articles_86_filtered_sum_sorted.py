import csv

input_file = 'Articles_86_filtered_sum.csv'
output_file = 'Articles_86_filtered_sum_sorted.csv'

with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)

# 按标题首字母排序（忽略大小写）
rows_sorted = sorted(rows, key=lambda x: x['Title'].strip().lower()[0] if x['Title'].strip() else '')

with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows_sorted)

print(f'排序完成，结果保存到：{output_file}')
