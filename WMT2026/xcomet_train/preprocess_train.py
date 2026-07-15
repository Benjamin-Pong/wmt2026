import pandas as pd
from collections import Counter
import argparse
from  difflib import SequenceMatcher
from pathlib import Path
import json
'''
This python script combines all xcomet's training data into the following 
Counter dictionaries should consist of the following 4 labels
1. No error
2. Minor error
3. Major error
4. Critical error

EVERY JSONL LINE SHOULD HAVE THE FOLLOWING FIELDS, MINIMALLY:

{'src': str
'mt': str
'ref': str (optional)
'annotations':List[Dict]
'lp': str}
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indic", help='path to directory of json files')
    parser.add_argument("--wmt", help='path to jsonl file containing wmt2020 to 2022 data')
    parser.add_argument("--demetr", help='path to demetr directory of jsonl files')
    parser.add_argument("--result", help='path to consolidated jsons')
    return parser.parse_args()

def reprocess_indicMT(indic_dir, res):
    lang_codes = {'Guj':'gu', 'Hin':'hi', 'Mal':'ml', 'Mar':'mr', 'Tam':'ta'}
    for file_path in Path(indic_dir).glob("*.jsonl"): #jsonl
        locale = file_path.name.split("\\")[-1].split("_")[0]
        lp=f'en-{lang_codes[locale]}'
 
        with open(file_path, "r", encoding="utf-8") as f:
            curr_file_data = [json.loads(line) for line in f]
        
        rename_keys = {'completion':'annotations', 'span_start_offset':'start', 'span_end_offset':'end', 'span_severity':'severity'}

        for line in curr_file_data:
            line = {rename_keys.get(k, k): v for k, v in line.items()}
            line['lp']=lp
            annotations = line['annotations']
            renamed_annotations=[]
            for annotation in annotations:
                renamed_annotation={rename_keys.get(k,k):v for k,v in annotation.items()}
                renamed_annotations.append(renamed_annotation)
            
            annotations = renamed_annotations
            line['annotations']=annotations
            res.write(json.dumps(line, ensure_ascii=False)+'\n')
            res.flush()
def reprocess_wmt(wmt_json, res):
    with open(wmt_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    for d in data:
        res.write(json.dumps(d, ensure_ascii=False)+'\n')
        res.flush()

def diff_spans(mt, pert):
    sm = SequenceMatcher(None, mt, pert, autojunk=False)
    return [(j1, j2) for tag, _, _, j1, j2 in sm.get_opcodes() if tag != "equal"]

def reprocess_demetr(demtr_dir, res):
    for file_path in Path(demtr_dir).glob("*.json"): #jsonl
        severity = file_path.name.split('\\')[-1].split("_")[0]
            
        with open(file_path, "r", encoding="utf-8") as f:
            curr_file_data = json.load(f)
        lang_map = {'french':'fr', 'czech':'cz', 'polish':'pl','chinese_simplified': 'zh', 'italian': 'it', 'german': 'de', 'russian': 'ru', 'spanish':'es', 'japanese':'jp', 'hindi':'hi' }
        for d in curr_file_data:
            lp=f'{d['lang_tag']}-en'
            list_error_span = diff_spans(d['mt_sent'], d['pert_sent'])
            if severity == "base":
                annotations=[]
            else:
                annotations= [{'start': tup[0], 'end':tup[1],'severity':severity} for tup in list_error_span]

            curr_res = {'src': d['src_sent'], 'mt': d['pert_sent'], 'ref': d['mt_sent'], 'annotations': annotations, 'lp':lp}
            res.write(json.dumps(curr_res, ensure_ascii=False) + '\n')
            res.flush()

if __name__ == "__main__":
    args = parse_args()

    with open(args.result, 'w', encoding='utf-8') as r:
        reprocess_wmt(args.wmt, r)
        reprocess_demetr(args.demetr, r)
        reprocess_indicMT(args.indic, r)
        




        



            

    
    


