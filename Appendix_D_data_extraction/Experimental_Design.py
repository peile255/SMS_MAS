import csv

# 输入输出文件路径
input_file = 'Extracted_Experimental_Design.csv'
output_file = 'Experimental_Design.csv'

# 分类函数
def classify_experimental_design(desc):
    desc_lower = desc.lower()

    quantitative_keywords = [
        'simulation', 'simulated', 'numerical', 'experiment', 'training', 'metric',
        'parameter', 'evaluation', 'hyperparameter', 'comparison', 'episode', 'dataset',
        'gpu', 'ram', 'tensorflow', 'trace-driven', 'environment'
    ]

    qualitative_keywords = [
        'conceptual', 'theoretical', 'philosophical', 'manual analysis', 'literature',
        'narrative', 'workshop', 'discussion'
    ]

    case_keywords = ['case study', 'case analysis', 'empirical case']

    is_quant = any(k in desc_lower for k in quantitative_keywords)
    is_qual = any(k in desc_lower for k in qualitative_keywords)
    is_case = any(k in desc_lower for k in case_keywords)

    if is_case and not (is_quant or is_qual):
        return 'Case-based'
    elif is_quant and is_qual:
        return 'Mixed'
    elif is_quant:
        return 'Quantitative'
    elif is_qual:
        return 'Qualitative'
    else:
        # 默认按定量归类（最常见）
        return 'Quantitative'

# 读取并写入文件
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'Experimental_Design'])
    writer.writeheader()

    for row in reader:
        category = classify_experimental_design(row['Experimental_Design'])
        writer.writerow({'ID': row['ID'], 'Experimental_Design': category})

print(f"实验设计分类完成，结果已写入：{output_file}")
