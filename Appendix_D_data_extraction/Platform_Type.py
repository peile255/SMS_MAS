import csv

# 输入输出文件
input_file = 'Extracted_Platform_Type.csv'
output_file = 'Platform_Type.csv'


# 分类规则函数（无 Other，强制归类）
def categorize(platform):
    platform_lower = platform.lower()

    if any(kw in platform_lower for kw in [
        'iot', '6g', 'network', 'edge', 'wireless', 'uav', 'infrastructure',
        'mec', 'vehicular', 'ntn', 'terrestrial', 'web service', 'scheduling'
    ]):
        return 'Infrastructure'
    elif any(
            kw in platform_lower for kw in ['blockchain', 'web 3.0', 'distributed system', 'web of things', 'quantum']):
        return 'Blockchain'
    elif any(kw in platform_lower for kw in ['simulation', 'unreal engine', 'cesium', 'game', 'video streaming']):
        return 'Simulation'
    elif any(kw in platform_lower for kw in
             ['social', 'community', 'xr users', 'virtual reality', 'vr headset', 'group']):
        return 'Social'
    elif any(kw in platform_lower for kw in ['industrial', 'manufacturing', 'iiot', 'robot', 'smart']):
        return 'Industrial'
    else:
        # 默认分类（如不匹配），可根据你对原始数据了解进行人工决策或默认选最贴近的
        return 'Simulation'  # 默认归为 Simulation


# 执行分类和写入新文件（仅保留 ID 和归类后的 Platform_Type）
with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'Platform_Type'])

    writer.writeheader()
    for row in reader:
        category = categorize(row['Platform_Type'])
        writer.writerow({'ID': row['ID'], 'Platform_Type': category})

print(f"分类完成，输出文件为：{output_file}")
