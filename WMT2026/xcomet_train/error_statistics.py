import argparse
import json
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
from comet import download_model, load_from_checkpoint
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import os
from huggingface_hub import whoami, get_token

print(load_dotenv(dotenv_path=env_path))   # False => file not found
print(repr(os.environ.get("HF_TOKEN"))[:12], "...")  # None => wrong var name
print(get_token())                          # what hub actually sees
print(whoami())                             # which account, or error



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to combined training set json')
    parser.add_argument('--error_statistics', help='output txt file containing error statistics')
    return parser.parse_args()

def process_annotations(json_line, xcomet_encoder, global_subword_label_distribution:Dict, lang_subword_label_distribution:Dict[str, Dict[str, int]], global_length:int, lang_length:Dict[str, Dict[str,int]]):
    lp = json_line['lp']
    annotations:List[Dict] = json_line['annotations']
    mt_sent:str = json_line['mt']
    ''''
    encoder_input = xcomet_encoder.tokenizer([mt_sent], #what is the input sample? - a list of strings
            truncation=True,
            max_length=xcomet_encoder.max_positions - 2) #encoder_input is an Encoding object of Huggingface Transformers Tokenizers
    '''
    labels = xcomet_encoder.subword_tokenize([mt_sent], annotations) #encoder_input[0] extracts the Encoding object for the sole sentence
    labels:List[List] = labels['input_labels']
    label_list = labels[0]
    print(label_list)
    '''
    how to compute label distribution?
    Two  ways:
    1. Global: Counter()
    2. Language-pair level {'en':Counter()..}

    denominator:
    1. Compute the total number of sub_tokens at the global level
    2. Compute total number of sub_tokens at the lp level
    '''
    global_subword_label_distribution.update(label_list)
    lang_subword_label_distribution[lp].update(label_list)
    length_subword_tokens = len(label_list)
    global_length+=length_subword_tokens
    lang_length[lp]+=length_subword_tokens

    return global_subword_label_distribution, lang_subword_label_distribution, global_length, lang_length



def error_statistics(global_subword_label_distribution, lang_subword_label_distribution, global_length, lang_length):
    for label in global_subword_label_distribution:
        continue
    pass



if __name__ == "__main__":
    args = parse_args()
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    #xcomet_tokenizer = model.encoder.tokenizer
    xcomet_encoder = model.encoder

    global_subword_label_distribution = Counter()
    lang_subword_label_distribution = defaultdict(Counter)
    lang_length = defaultdict(int)
    global_length = 0

    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        #type cast all start and end indices to integers, some are strings
        #udata = []
        for d in data:
            if d['annotations']!=[]: #edge case, no error
                for error_span in d['annotations']:
                    for field in ("start", "end"):
                        error_span[field]=int(error_span[field])
            #udata.append(d)
        #data=udata

    for d in data[0:2]:
        global_subword_label_distribution, lang_subword_label_distribution, global_length, lang_length = process_annotations(d, xcomet_encoder, global_subword_label_distribution, lang_subword_label_distribution, global_length, lang_length)
        
    
    print(global_subword_label_distribution)