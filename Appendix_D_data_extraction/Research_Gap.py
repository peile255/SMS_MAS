import csv

def classify_gap(text):
    text_lower = text.lower()

    if any(k in text_lower for k in [
        "real-world validation", "real-world deployment", "deployment", "cross-platform", "live-user", "sim2real", "field test"
    ]):
        return "Validation"
    elif any(k in text_lower for k in [
        "framework", "architecture", "design", "system structure", "standardized definition", "unified framework"
    ]):
        return "System Design"
    elif any(k in text_lower for k in [
        "multi-agent", "npc", "mas", "agents", "interaction", "cooperation", "collaborative", "intelligent npc"
    ]):
        return "Multi-Agent Interaction"
    elif any(k in text_lower for k in [
        "privacy", "security", "attack", "oracle", "tampering", "cybersecurity", "consensus", "encryption", "integrity"
    ]):
        return "Security"
    elif any(k in text_lower for k in [
        "integration", "combine", "incorporate", "fusion", "merge", "iot", "blockchain", "web 3.0", "generative ai", "interoperability"
    ]):
        return "Technology Integration"
    elif any(k in text_lower for k in [
        "dynamic", "adaptability", "real-time", "time-varying", "changing environment", "temporal"
    ]):
        return "Dynamic Adaptability"
    elif any(k in text_lower for k in [
        "model", "algorithm", "prediction", "optimization", "training", "reinforcement learning", "robustness", "performance", "learning", "exploration"
    ]):
        return "Models"
    else:
        return "Models"  # 默认分类，避免遗漏

input_file = 'Extracted_Research_Gap.csv'
output_file = 'Research_Gap.csv'

with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'Research_Gap'])
    writer.writeheader()

    for row in reader:
        category = classify_gap(row['Research_Gap'])
        writer.writerow({'ID': row['ID'], 'Research_Gap': category})

print(f"Research_Gap.csv 已生成，类别归类完成。")
