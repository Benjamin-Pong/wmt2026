'''
This script computes the metrics for summary statistics and callibration
Different levels of evaluation: Correlation between confidence and entrop
'''

import json
import torch
import torch.distributions as td
import torch.nn.functional as F
import argparse
from typing import List, Dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_json", help='path to pred_json')
    parser.add_argument("--output", help='path to updated pred_json with additional metrics')
    return parser.parse_args()
def compute_subword_entropy(subword_probs):
    '''
    Input args: subword_probs
    Output: produces a tensor containing scalar outputs 
    '''
    entropy = td.Categorical(probs=subword_probs)
    entropy = entropy.entropy()
    return entropy

def compute_average_shannon_entropy(curr_span_probabilities, eps: float = 1e-12) -> torch.Tensor:
    '''
    Computes entropy value per subword token, and returns the mean over all tokens
    '''
    log_probs = torch.log(curr_span_probabilities + eps)
    entropy = -torch.sum(curr_span_probabilities * log_probs, dim=-1)
    return entropy.mean()

def compute_continuous(subword_probs):
    continuous_metric = (1- subword_probs[:, -3:].sum(axis=-1)).T
    return continuous_metric

def compute_all_span_entropy(subword_probs, error_span:List[Dict]):
    '''
    This method computes the  entropy for each predicted error span, similar to xcomet's confidence per span
    Input arguments include subword_probs and error_span per system
    It updates each span prediction (dictionary) with span entropy
    '''
    updated_error_span = []
    for curr_span in error_span:
        print(curr_span)
        print(curr_span.keys())
        curr_span_probs = subword_probs[curr_span['start']:curr_span['end'], :]
        print(subword_probs.shape)
        curr_span_entropy = compute_average_shannon_entropy(curr_span_probs)
        curr_span['span_entropy'] = curr_span_entropy.item()
        if curr_span_entropy.item():
            print(curr_span_entropy)
        else:
            "does not exist"
            break
    
        updated_error_span.append(curr_span)
    return updated_error_span

def compute_average_confidence(subword_probs:torch.Tensor) -> torch.Tensor:
    '''
    

    '''
    get_max = torch.max(subword_probs, dim=-1).values
    average_conf = get_max.mean()
    return average_conf


def compute_metrics(data, output):
    '''
    input args
    Computes all relevant metrics 
    '''
    with open(data, 'r', encoding='utf-8') as f:
        data = [json.loads(f.readline())]
    with open(output, 'w', encoding='utf-8') as o:
        for d in data:
            scores_dict = d['scores']
            for system in scores_dict:
                #prediction =  scores_dict[system][]
                
                subword_probs = torch.tensor(scores_dict[system]['subword_probs']) #(batch size, seq_length, 4)
                bias_corrected_logits = torch.tensor(scores_dict[system]['logits'])
                entropy = compute_subword_entropy(subword_probs)
                scores_dict[system]['subword_entropy']=entropy.tolist() #store entropy as a metric to json

                #continuous = compute_continuous(subword_probs)
                #scores_dict[system]['continuous']=continuous.tolist()

                error_span = scores_dict[system]['error_span']
                updated_error_span_with_span_entropy = compute_all_span_entropy(subword_probs, error_span)
                scores_dict[system]['error_span'] = updated_error_span_with_span_entropy

                #initialize average entropy over all subtokens in a mt segment 
                average_entropy = compute_average_shannon_entropy(subword_probs)
                scores_dict[system]['average_entropy']=average_entropy.item()

                #initialize average confidence over all subtokens in a mt segment
                average_confidence = compute_average_confidence(subword_probs)
                scores_dict[system]['average_confidence']=average_confidence.item()

            d['scores'] = scores_dict
            o.write(json.dumps(d)+'\n')

def main():
    args = parse_args()
    compute_metrics(args.pred_json, args.output)
    

if __name__ == '__main__':
    main()
    

        


