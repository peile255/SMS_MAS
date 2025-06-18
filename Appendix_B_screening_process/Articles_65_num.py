import csv

input_file = 'Articles_65.csv'
output_file = 'Articles_65_num.csv'

with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)

# 重新给Article_Number编号，格式为三位数，从001开始
for i, row in enumerate(rows, start=1):
    row['Article_Number'] = f'{i:03d}'

with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'重新编号完成，结果保存到：{output_file}')
