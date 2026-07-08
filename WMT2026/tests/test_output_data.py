
from pathlib import Path
from dotenv import load_dotenv
import pytest

import os

import json
import torch


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
        tgt_lang = line['doc_id'].split('_')[0].split('-')[1]
        domain_name = line['doc_id'].split('#')[1].strip('_')

        all_src_langs.add(src_lang)
        all_tgt_langs.add(tgt_lang)
        if tgt_lang=='CN':
            print(relevant_text)
            print(line['scores'])
    print(all_src_langs)
    print(all_tgt_langs)
    print(total_lines)

def inspect_shapes(human_eval):
    data = human_eval[-1]
    logits = torch.tensor(data['scores']['refA']['logits']).shape
    subword_probs = torch.tensor(data['scores']['refA']['subword_probs']).shape
    return logits==subword_probs

def inspect_tokens(human_eval):
    data = human_eval[-1]
    pass

def test_document_string(human_eval):
    line=human_eval[-1]
    src_lang = line['doc_id'].split('_')[0].split('-')[0]
    tgt_lang = line['doc_id'].split('_')[0].split('-')[1]
    domain_name = line['doc_id'].split('#')[1].strip('_')
    print(src_lang, tgt_lang, domain_name)

def test_size(human, results):
    assert len(human) == len(results)

def test_json_structure():
    pass
    
if __name__ == "__main__":
    results=r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl"
    human_eval = r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\wmt25-genmt-humeval.jsonl"
    with open(results, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    #print(human_eval)
    with open(human_eval, 'r', encoding='utf-8') as g:
        human = [json.loads(line) for line in g]
    #print(inspect_shapes(results))
    print(test_document_string(results))
    print(test_size(human, results))
