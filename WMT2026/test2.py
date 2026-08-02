import json

file = r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\xcometlogadj.0.5.zh_ja.refined.jsonl"

with open (file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]


for line in data:
    print(len(line['task1_pred']))
    for system in line['task1_pred']:
        print(line['task1_pred'][system].keys())
        print(line['task1_pred'][system]['errors'])
        print(line['task1_pred'][system]['corrected_topk_spans'])
        print(line['task1_pred'][system]['refined_errors'])
        print(line['task1_pred'][system]['reasoning_trace'])
        break
        
