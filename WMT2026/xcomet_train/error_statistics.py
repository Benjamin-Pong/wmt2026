import argparse
import json
from comet import download_model, load_from_checkpoint
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to combined training set json')
    parser.add_argument('--error_statistics', help='output txt file containing error statistics')
    return parser.parse_args()

def process_annotations(json_line, xcomet_tokenizer, subword_label_distribution):
    annotations:List[Dict] = json_line['annotations']
    mt_sent:str = json_line['mt']
    encoder_input = xcomet_tokenizer([mt_sent], #what is sample?
            truncation=True,
            max_length=xcomet_tokenizer.max_positions - 2) #encoder_input is an Encoding object of Huggingface Transformers Tokenizers
    
    labels = xcomet_tokenizer.subword_tokenize(encoder_input[0], annotations) #encoder_input[0] extracts the Encoding object for the sole sentence
    labels:List[List] = labels['input_labels']
    label_list = labels[0]
    '''
    how to compute label distribution?
    Two  ways:
    1. Global: Counter()
    2. Language level {'en':Counter()..}
    '''

    return subword_label_distribution

if __name__ == "__main__":
    args = parse_args()
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    xcomet_tokenizer = model.encoder.tokenizer

    subword_label_distribution = Counter()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    for d in data:
        process_annotations(d, xcomet_tokenizer, subword_label_distribution)