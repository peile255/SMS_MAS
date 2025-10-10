# 重新生成标准化CSV文件
import csv

with open('titles.csv', 'r', encoding='utf-8') as infile, \
        open('titles_clean.csv', 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)  # 强制所有字段加引号

    for row in reader:
        writer.writerow(row)