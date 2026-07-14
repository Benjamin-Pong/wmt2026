
from pathlib import Path
from dotenv import load_dotenv
import pytest

import os

import json
import torch

@pytest.fixture
def load_gold_json():
    human_eval = r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\wmt25-genmt-humeval.jsonl"
    #print(human_eval)
    with open(human_eval, 'r', encoding='utf-8') as g:
        human = [json.loads(line) for line in g]
    return human
    
@pytest.fixture
def load_results_json(): #just need to test 1 json
    results=r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl"
    with open(results, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    return results

@pytest.fixture
def load_updated_metrics_json():
    results=r"C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\metrics\xcomet.wmt2025esa.full.result.updated.jsonl"
    with open(results, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    return results

'''
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
'''

def test_gold_logits_subword_shape_match(load_gold_json):
    gold_data = load_gold_json[-1]
    gold_logits = torch.tensor(gold_data['scores']['refA']['logits']).shape
    gold_subword_probs = torch.tensor(gold_data['scores']['refA']['subword_probs']).shape
    #gold_tokens = torch.tensor(gold_data['scores']['tokens']).shape
    assert gold_logits==gold_subword_probs

def test_results_logits_subword_shape_match(load_results_json):
    result_data = load_results_json[-1]
    result_logits = torch.tensor(result_data['scores']['refA']['logits']).shape
    result_subword_probs = torch.tensor(result_data['scores']['refA']['subword_probs']).shape
    #result_tokens = torch.tensor(result_data['scores']['refA']['tokens']).shape
    assert result_logits==result_subword_probs


def test_document_string(human_eval):
    line=human_eval[-1]
    src_lang = line['doc_id'].split('_')[0].split('-')[0]
    tgt_lang = line['doc_id'].split('_')[0].split('-')[1]
    domain_name = line['doc_id'].split('#')[1].strip('_')
    print(src_lang, tgt_lang, domain_name)

def test_size(load_gold_json, load_results_json):

    assert len(load_results_json) == len(load_gold_json)


def test_json_structure():
    pass

def test_updated_metrics_fields(load_updated_metrics_json):
    '''
    tests that json contains the additional metrics fields computed.
    '''
    error_span_data = load_updated_metrics_json[-1]['scores']['refA']['error_span']
    assert 'span_entropy' in error_span_data.keys()



    
