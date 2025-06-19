import csv

input_file = 'Extracted_System_Architecture.csv'
output_file = 'System_Architecture.csv'

def classify_architecture(desc):
    desc_lower = desc.lower()

    # 1. Three-layer
    if 'three-layer' in desc_lower or ('thing layer' in desc_lower and 'cloud' in desc_lower):
        return 'Three-layer'

    # 2. Multi-agent
    if any(k in desc_lower for k in ['multi-agent', 'multiagent', 'mas', 'agent']):
        return 'Multi-agent'

    # 3. Blockchain-based
    if any(k in desc_lower for k in ['blockchain', 'oracle', 'consortium chain', 'main chain', 'relay chain', 'minor chain']):
        return 'Blockchain-based'

    # 4. Digital Twin
    if any(k in desc_lower for k in ['digital twin', 'dt layer', 'hdt']):
        return 'Digital Twin'

    # 5. Edge-Cloud collaborative
    if any(k in desc_lower for k in ['edge', 'fog', 'cloud']) and ('collaborative' in desc_lower or 'collaboration' in desc_lower):
        return 'Edge-Cloud'

    # 6. Hierarchical / Multi-layer / Multi-tier / Multi-stage (不包含前三类)
    if any(k in desc_lower for k in ['hierarchical', 'multi-layer', 'multi-tier', 'multi-stage', 'three-stage']):
        return 'Hierarchical / Multi-layer'

    # 7. Other
    return 'Other'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'System_Architecture'])
    writer.writeheader()

    for row in reader:
        desc = row['System_Architecture']
        category = classify_architecture(desc)
        writer.writerow({'ID': row['ID'], 'System_Architecture': category})

print(f"系统结构分类完成，结果已写入：{output_file}")
