original = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\wmt25-genmt-humeval.jsonl'
xcomet = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometmetriclayer.wmt2025esa.full.result.jsonl'

import json
import torch

with open(original, 'r', encoding='utf-8') as o:
    ori = [json.loads(line) for line in o]
with open(xcomet, 'r', encoding='utf-8') as xc:
    xcom = [json.loads(line) for line in xc]

print(len(ori))
print(xcom[-1]['source_segment'])
print(ori[273]['src_text'])

print(torch.tensor(xcom[0]['scores']['refA']['logits']).shape)
print(torch.tensor(xcom[0]['scores']['refA']['subword_probs']).shape)
#logits and subwords have different dimensions. Different seq_length, why??