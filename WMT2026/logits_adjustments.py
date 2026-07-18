'''
Program to generate summary statistics / exploratory data analyses
input into the program would be the output of 
'''

import pandas as pd
import argparse
from collections import Counter
import torch
from typing import List, Dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_json', help='the path to pred_json, with updated metrics')
    parser.add_arguemtn()
    return parser.parse_args()


def global_logits_adjustment(global_error_stats:List, logits_batch):
    '''
    This function passes in the logits for each target segment's , 
    adjusts logits using logits offsets
    '''
    ln_global_error_stats = torch.log(torch.tensor(global_error_stats))
    logits_batch = ln_global_error_stats
    

    
    
def compute_entropy_confidence_correl(results):


    


