'''
Python script creates tsv from result.jsonl
'''
import json
from collections import Counter
import pandas as pd
import argparse





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_json', required=True)
    parser.add_argument('--prediction_tsv', required=True)
    return parser.parse_args()

def unpack_error_span(error_spans):
    '''
    input is a list of error_spans
    error_span is a dictionary
    outputs start indices, end indices, and severity labels as individual lists
        - These lists contain their respective information in an enumerated string format

    '''
    start_indices = []
    end_indices = []
    error_types = []
    for error_span in error_spans:
        start_indices.append(str(error_span['start']))
        end_indices.append(str(error_span['end']))
        error_types.append(error_span['severity'])
    
    return ' '.join(start_indices), ' '.join(end_indices), ' '.join(error_types)



def extract_values(instance, counters_by_lp):
    '''
    input is a jsonl
    produces every systems' prediction metadata into a tsv row
    '''
    source_lang = instance['source_lang']
    target_lang = instance['target_lang']
    source_text = instance['source_segment']
    doc_id=instance['doc_id']
    domain = instance['domain']
    lp = instance['doc_id'].split('_')[0]

    doc_data = {'doc_id': doc_id,'source_lang': source_lang, 'target_lang': target_lang, 'source_segment': source_text, 'domain': domain}
    

    for score in instance['scores'].values():
        #doc_id = score['doc_id']
        #doc_data['doc_id']=doc_id
        system_id = score['system_id']
        doc_data['system_id']=system_id
        target_segment = score['prediction']
        doc_data['hypothesis_segment']=target_segment
        method = 'ESA'
        doc_data['method']=method
        overall = score['segment_score']
        doc_data['overall']=overall
        
        
       
        #lp = score['doc_id'].split('_')[0]
        if lp not in counters_by_lp:
            counters_by_lp[lp] = 0
        segment_id = counters_by_lp[lp]
        counters_by_lp[lp]+=1

        start_indices, end_indices, error_types = unpack_error_span(score['error_span'])
        doc_data['segment_id']= segment_id
        doc_data['start_indices']= start_indices
        doc_data['end_indices']= end_indices
        doc_data['error_types']= error_types

    return doc_data, counters_by_lp


def main(TSV_FIELDS_RELEASE, results_json, output_file_tsv):
    #results_json = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl'
    with open(results_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    rows=[]
    
    counters_by_lp = {}
    for instance in data:
        doc_data, counters_by_lp_ = extract_values(instance, counters_by_lp)
        counters_by_lp = counters_by_lp_
        rows.append({field: doc_data.get(field, "") for field in TSV_FIELDS_RELEASE})
    
    df = pd.DataFrame(rows, columns=TSV_FIELDS_RELEASE)
    df.to_csv(output_file_tsv, sep='\t', index=False)
        

if __name__ == "__main__":
    TSV_FIELDS_RELEASE = [
    "doc_id",
    "segment_id",
    "source_lang",
    "target_lang",
    "set_id",
    "system_id",
    "source_segment",
    "hypothesis_segment",
    "domain_name",
    "method",
    "start_indices", 
    "end_indices", 
    "error_types"
]
    args = parse_args()
    main(TSV_FIELDS_RELEASE, args.results_json, args.prediction_tsv)








    
