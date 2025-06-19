import csv

# 输入输出文件名
input_file = 'Extracted_Challenge_Type.csv'
output_file = 'Challenge_Type.csv'

# 分类函数
def classify_challenge(desc):
    desc_lower = desc.lower()

    if any(k in desc_lower for k in [
        'latency', 'bandwidth', 'network', 'transmission', 'scheduling', 'throughput',
        'rsu', 'communication', 'qos', 'e2e', 'routing', 'packet', 'channel', 'congestion'
    ]):
        return 'Network'
    elif any(k in desc_lower for k in [
        'computation', 'resource', 'task', 'mec', 'offloading', 'load', 'scalability',
        'optimization', 'overhead', 'storage', 'response time', 'cloud', 'cache',
        'cpu', 'gpu', 'qubit'
    ]):
        return 'Resource Management'
    elif any(k in desc_lower for k in [
        'multi-agent', 'mas', 'coordination', 'cooperation', 'collaboration', 'negotiation',
        'consensus', 'conflict', 'agent', 'herding'
    ]):
        return 'Multi-Agent Collaboration'
    elif any(k in desc_lower for k in [
        'metaverse', 'avatar', 'immersive', 'xr', 'vr', 'ar', 'hmd', 'digital twin',
        'virtual', 'cyberspace', 'experience', 'qoe', 'presence', 'hologram'
    ]):
        return 'Metaverse'
    elif any(k in desc_lower for k in [
        'model', 'training', 'learning', 'ai', 'fidelity', 'accuracy', 'prediction',
        'data', 'dataset', 'asymmetry', 'semantic', 'knowledge', 'ontology', 'simulation model'
    ]):
        return 'Models'
    elif any(k in desc_lower for k in [
        'blockchain', 'security', 'privacy', 'encryption', 'tampering', 'oracle', 'attack',
        'anonymity', 'integrity', 'consensus', 'proof', 'pseudonym', 'cybersecurity'
    ]):
        return 'Security'
    else:
        return 'Models'  # 默认归为 Data and Models，防漏分类

# 文件读写
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['ID', 'Challenge_Type'])
    writer.writeheader()

    for row in reader:
        category = classify_challenge(row['Challenge_Type'])
        writer.writerow({'ID': row['ID'], 'Challenge_Type': category})

print(f"挑战类型分类完成，结果已写入：{output_file}")
