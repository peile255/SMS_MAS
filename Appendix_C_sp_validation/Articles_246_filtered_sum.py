import csv

input_file = 'Articles_332_score_sum.csv'
output_file = 'Articles_246_filtered_sum.csv'

# 允许的sum值列表
allowed_sums = {0, 0.5, 1}

with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        try:
            sum_value = float(row['Sum'])
            if sum_value in allowed_sums:
                writer.writerow(row)
        except ValueError:
            # 遇到无法转换为float的sum值则跳过
            continue

print(f'筛选完成，结果保存到：{output_file}')
