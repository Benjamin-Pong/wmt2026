'''
This script computes the metrics for summary statistics and callibration
Different levels of evaluation: Correlation between confidence and entrop
'''


input_file = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometmetriclayer.wmt2025esa.full.result.jsonl'

import json
import torch
import torch.distributions as td
from comet import download_model, load_from_checkpoint
import torch.nn.functional as F


def compute_subword_entropy(subword_probs):
    '''
    Input args: subword_probs
    Output: produces a tensor containing scalar outputs 
    '''
    entropy = td.Categorical(probs=subword_probs)
    entropy = entropy.entropy()
    return entropy

def compute_shannon_entropy(curr_span_probabilities, eps: float = 1e-12):
    log_probs = torch.log(curr_span_probabilities + eps)
    entropy = -torch.sum(curr_span_probabilities * log_probs, dim=-1)
    return entropy

def compute_continuous(subword_probs):
    continuous_metric = (1- subword_probs[:,:,:, -3:].sum(axis=-1)).T
    return continuous_metric

def compute_all_span_entropy(subword_probs, error_span):
    '''
    Input arguments include subword_probs and error_span per system
    It updates each span prediction (dictionary) with span entropy
    '''
    updated_error_span = []
    for curr_span in error_span:
        curr_span_probs = subword_probs[:, curr_span['start_i']:curr_span['end_i'], :]
        curr_span_entropy = compute_shannon_entropy(curr_span_probs)
        curr_span['span_entropy'] = curr_span_entropy
        updated_error_span.append(curr_span)
    return updated_error_span

def compute_logits_adjustment(logits):

    pass

def main(data, output, xcomet_tokenizer):
    '''
    input args
    Computes all relevant metrics 
    '''
    for d in data:
        scores_dict = d['scores']
        for system in scores_dict:
            #prediction =  scores_dict[system][]
            
            subword_probs = torch.tensor(scores_dict[system]['subword_probs']) #(batch size, seq_length, 4)
            bias_corrected_logits = torch.tensor(scores_dict[system]['logits'])
            entropy = compute_subword_entropy(subword_probs)
            scores_dict[system]['subword_entropy']=entropy.tolist() #store entropy as a metric to json

            continuous = compute_continuous(subword_probs)
            scores_dict[system]['continuous']=continuous

            error_span = scores_dict[system]['error_span']
            updated_error_span_with_span_entropy = compute_all_span_entropy(subword_probs, error_span)
            scores_dict[system]['error_span'] = updated_error_span_with_span_entropy


        d['scores'] = scores_dict
        output.write(json.dumps(d)+'\n')
    pass
if __name__ == '__main__':
    model_path = download_model("Unbabel/XCOMET-XL")
    
    xcomet = load_from_checkpoint(model_path)
    #xcomet_tokenizer = xcomet.encoder.tokenizer
    main()

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]


        


