
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help='Path to input file')
    return parser.parse_args()

def get_average_confidence_entropy(data):
    '''
    This function computes the correlation between span level entropy and 
    Recall with average entropy across
    Hypothesis: More entropic error spans tend to be true errors

    if span errors exist, rank the span error hypotheses according to their entropies
    '''
    confidence=[]
    entropy=[]
    for d in data:
        scores = d['scores']
        for system in scores:
            confidence.append(scores[system]['average_confidence'])
            entropy.append(scores[system]['average_entropy'])
    return confidence, entropy

def plot_distribution(metric):
    plt.hist(metric, bins=30)
    plt.xlabel(f"Average {metric}")
    plt.savefig(f"{metric}.png", dpi=150)


if __name__ == "__main__":
    args = parse_args()
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    confidence, entropy = get_average_confidence_entropy(data)
    plot_distribution(confidence)
    plot_distribution(entropy)
    