'''
This script computes the metrics for statistical analysis and evaluation of outcome
Different levels of evaluation: Correlation between confidence and entrop
'''


input_file = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometmetriclayer.wmt2025esa.full.result.jsonl'

import json
import torch
import torch.distributions as td
from comet import download_model, load_from_checkpoint

with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(len(data))
print(data[0]['scores']['refA'].keys())

#check that dimensions of logits and subword_probs are the same
logits=torch.tensor(data[0]['scores']['refA']['logits'])    
subword_probs = torch.tensor(data[0]['scores']['refA']['subword_probs'])
seq_len = subword_probs.shape[1]
print(seq_len)
print(logits[:, :seq_len,:].shape)
print(subword_probs.shape)
print(logits.shape==subword_probs.shape)


def compute_entropy(subword_probs):
    '''
    Input args: subword_probs
    Output: produces a tensor containing scalar outputs 
    '''
    entropy = td.Categorical(probs=subword_probs)
    entropy = entropy.entropy()
    return entropy
def compute_continuous(subword_probs):
    continuous_metric = (1- subword_probs[:,:,:, -3:].sum(axis=-1)).T
    return continuous_metric

    

def main(data, output, xcomet_tokenizer):
    '''
    input args
    Computes all relevant metrics 
    '''
    for d in data:
        scores_dict = d['scores']
        for system in scores_dict:
            prediction =  scores_dict[system][]
            subword_probs = torch.tensor(scores_dict[system]['subword_probs']) #(batch size, seq_length, 4)
            #subword_probs = trim_subword_probs(subword_probs, prediction, xcomet_tokenizer)
            entropy = compute_entropy(subword_probs)
            scores_dict[system]['entropy']=entropy.tolist() #store entropy as a metric to json
        d['scores'] = scores_dict
        output.write(json.dumps(d)+'\n')
    pass
if __name__ == '__main__':
    model_path = download_model("Unbabel/XCOMET-XL")
    
    xcomet = load_from_checkpoint(model_path)
    xcomet_tokenizer = xcomet.encoder.tokenizer
    main()
    

        


