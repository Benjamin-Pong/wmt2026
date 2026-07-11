'''
Program to generate summary statistics / exploratory data analyses
input into the program would be the output of 
'''

import pandas as pd
import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_json', help='the path to pred_json, with updated metrics')
    return parser.parse_args()
def compute_label_distribution(results):
    pass


def compute_entropy_confidence_correl(results):


    


