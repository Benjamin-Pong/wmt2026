import pandas as pd
from collections import Counter
import argparse
import difflib
from pathlib import Path
import json
'''
This python script combines all xcomet's training data into the following 
Counter dictionaries should consist of the following 4 labels
1. No error
2. Minor error
3. Major error
4. Critical error

EVERY JSONL LINE SHOULD HAVE THE FOLLOWING FIELDS MINIMALLY
{'src': str
'mt': str
'annotations':List[Dict]
'lp': str}
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indic", help='path to directory of json files')
    parser.add_argument("--wmtmqm20202022", help='path to jsonl file containing wmt2020 to 2022 data')
    parser.add_argument("--demeter", help='path to demetr directory of jsonl files')
    parser.add_argument("--output", help='path to consolidated jsons')

def reprocess_indicMT(indic_dir, res):
    lang_codes = {'Guj':'gu', 'Hin':'hi', 'Mal':'ml', 'Mar':'mr', 'Tam':'ta'}
    for file_path in Path(indic_dir).glob("*.jsonl"): #jsonl
        locale = file_path.split("_")[0].lower()
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
            

    
    


