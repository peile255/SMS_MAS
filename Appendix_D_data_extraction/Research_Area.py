import csv

# 输入输出文件
input_file = 'Extracted_Research_Area.csv'
output_file = 'Research_Area.csv'

# 分类规则（固定5类）
def classify_research_area(area):
    area_lower = area.lower()

    if any(keyword in area_lower for keyword in [
        'artificial intelligence', 'reinforcement learning', 'machine learning', 'data-driven', 'ai safety'
    ]):
        return 'Artificial Intelligence'
    elif any(keyword in area_lower for keyword in [
        'blockchain', 'secure', 'contract theory', 'web 3.0', 'distributed'
    ]):
        return 'Blockchain'
    elif any(keyword in area_lower for keyword in [
        'simulation', 'virtual reality', 'digital twin', 'military decision', 'game', 'virtual simulation', 'airsim'
    ]):
        return 'Simulation'
    elif any(keyword in area_lower for keyword in [
        'social', 'opinion', 'loyalty', 'human-computer interaction', 'community', 'hci', 'group', 'library', 'information'
    ]):
        return 'Social Computing'
    elif any(keyword in area_lower for keyword in [
        'vehicular', 'manufacturing', 'robotic', 'industrial', 'resource allocation', 'edge computing', 'cyber-physical', 'iot', 'network', 'infrastructure'
    ]):
        return 'Industrial Systems'
    else:
        # 默认归类到 AI，避免 Other
        return 'Artificial Intelligence'

# 处理 CSV
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'Research_Area'])

    writer.writeheader()
    for row in reader:
        classified_area = classify_research_area(row['Research_Area'])
        writer.writerow({'ID': row['ID'], 'Research_Area': classified_area})

print(f"分类完成，结果已保存至：{output_file}")
