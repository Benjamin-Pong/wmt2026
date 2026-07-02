
from pathlib import Path
from dotenv import load_dotenv

import os

import json




def get_systems_evaluated(line):
    return set(line['scores'].keys())
def score(human_eval):
    all_src_langs = set()
    all_tgt_langs = set()
    total_lines = len(human_eval)
    for line in human_eval:
        src_text = line['src_text']
        evaluated_systems = get_systems_evaluated(line)
        relevant_text = {k:v for k,v in line['tgt_text'].items() if k in evaluated_systems}
        src_lang = line['doc_id'].split('_')[0].split('-')[0]
        tgt_lang = line['doc_id'].split('_')[1].split('-')[0]
        all_src_langs.add(src_lang)
        all_tgt_langs.add(tgt_lang)
        if tgt_lang=='CN':
            print(relevant_text)
            print(line['scores'])
    print(all_src_langs)
    print(all_tgt_langs)
    print(total_lines)

if __name__ == "__main__":
    human_eval_file=r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\wmt25-genmt-humeval.jsonl"
    with open(human_eval_file, 'r', encoding='utf-8') as f:
      human_eval = [json.loads(line) for line in f]
    #print(human_eval)
    score(human_eval)
