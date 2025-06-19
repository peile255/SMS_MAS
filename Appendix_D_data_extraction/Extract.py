import csv

# 输入输出文件名
input_file = 'Articles_78.csv'
output_file = 'Extracted_Source_Name.csv'

# 要提取的列名
fields_to_extract = ['ID', 'Source_Name']

# 执行提取
with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=fields_to_extract)

    writer.writeheader()
    for row in reader:
        writer.writerow({field: row[field] for field in fields_to_extract})

print(f"提取完成，新文件保存为：{output_file}")
