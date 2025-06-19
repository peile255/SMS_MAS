import csv

def classify_evaluation_limit(text):
    text_lower = text.lower()

    if any(k in text_lower for k in [
        "simulation", "no real-world", "real-world deployment", "real-world validation", "physical experiment", "controlled environment"
    ]):
        return "Validation Environment"
    elif any(k in text_lower for k in [
        "scalability", "large-scale", "extreme", "complex service scenarios", "high traffic", "edge cases"
    ]):
        return "Scalability"
    elif any(k in text_lower for k in [
        "specific model", "limited to", "generalizability", "fixed parameters", "turtlebot", "female-only", "domain-specific", "sample bias"
    ]):
        return "Generalizability"
    elif any(k in text_lower for k in [
        "human judgment", "subjective", "reviewer", "ontology construct", "philosophical"
    ]):
        return "Subjectivity"
    elif any(k in text_lower for k in [
        "private dataset", "data set not publicly available", "data availability", "data limitations"
    ]):
        return "Data Constraints"
    else:
        return "Validation Environment"  # 默认归类

input_file = 'Extracted_Evaluation_Limitations.csv'
output_file = 'Evaluation_Limitations.csv'

with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'Evaluation_Limitations'])
    writer.writeheader()

    for row in reader:
        category = classify_evaluation_limit(row['Evaluation_Limitations'])
        writer.writerow({'ID': row['ID'], 'Evaluation_Limitations': category})

print("✅ 新文件 Evaluation_Limitations.csv 已生成并完成分类。")
