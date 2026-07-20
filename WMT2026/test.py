original = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\wmt25-genmt-humeval.jsonl'
xcomet = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometlogitsadj.base.jsonl'

import json
import torch
from collections import Counter
'''
with open(original, 'r', encoding='utf-8') as o:
    ori = [json.loads(line) for line in o]
'''
with open(xcomet, 'r', encoding='utf-8') as xc:
    xcom = [json.loads(line) for line in xc]


counts = Counter()
for d in xcom:
    for system in d['scores']:
        for error_span in d['error_span']:
            counts[error_span['severity']]+=1

print(counts)
