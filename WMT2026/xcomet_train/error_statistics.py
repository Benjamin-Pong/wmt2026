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

def load_indic(indic_dir):


def reprocess_indicMT(indic_dir):
    '''
    Combines all indic data into one unified jsonl file, whose format per line is consistent with Ricardo's
    indic mqm has 
    1. default
    2. low
    3. medium
    4. high
    5. very high
    
    mapping: 
    1. default -> No error
    2. low -> Minor error
    3. medium -> Major error
    4. high -> Critical error
    5. very high -> Critical error


    '''

    
    for file_path in Path(indic_dir).glob("*.jsonl"): #jsonl
        locale = file_path.split("_")[0].lower()
        

        with open(file_path, "r", encoding="utf-8") as f:
            data[file_path.name] = json.load(f)
    return data
    pass


