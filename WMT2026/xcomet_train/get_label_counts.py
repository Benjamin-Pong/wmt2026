import pandas as pd
from collections import Counter
import argparse

'''
Counter dictionaries should consist of the following 4 labels
1. No error
2. Minor error
3. Major error
4. Critical error
'''
def get_indic_labels():
    '''
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

    Problem with indic MQM: No span level information, so not able to compute probabilities/frequencies of each label.
    '''
