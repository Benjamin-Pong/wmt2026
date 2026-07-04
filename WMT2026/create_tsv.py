'''
Python script creates tsv from result.jsonl
'''
import json

results = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl'
with open(results, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]


def unpack_error_span(error_spans):
    '''
    input is a list of error_spans
    error_span is a dictionary
    '''
    for error_span in error_spans:

def extract_values(instance):
    '''
    input is a jsonl
    produces every systems' prediction metadata into a tsv row
    '''
    source_lang = instance['source_lang']
    target_lang = instance['target_lang']
    source_text = instance['source_segment']
    

    for score in instance['scores']:
        doc_id = score['doc_id']
        system_id = score['system_id']
        target_segment = score['prediction']
        method = 'ESA'
        overall = score['segment_score']
        domain = score['doc_id'].split('#')[1].strip('_')
        segment_id = ?

        unpack_error_span(score['error_span'])

 [{"text": "有信心", "confidence": 0.3548550307750702, "severity": "major", "start": 15, "end": 18}
     'segment_score':model_output.scores[0], 'system_score':model_output.system_score, 'error_span':model_output.metadata.error_spans[0], 'logits': model_output.metadata.logits, 'subword_probs': model_output.metadata.subword_probs, 'human_score':human_score_per_system}


    
