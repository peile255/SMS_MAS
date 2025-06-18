import csv

input_file = 'Articles_332_score.csv'
output_file = 'Articles_332_score_sum.csv'

with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    # 新字段名
    fieldnames = reader.fieldnames + ['Sum']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        # 计算ChatGPT和DeepSeek的分数和
        chatgpt_score = float(row['ChatGPT'])
        deepseek_score = float(row['DeepSeek'])
        row['Sum'] = chatgpt_score + deepseek_score

        writer.writerow(row)

print(f'输出完成，文件保存为：{output_file}')
