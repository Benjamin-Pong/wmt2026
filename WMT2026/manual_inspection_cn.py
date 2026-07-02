'''

'''
import json

xcometlayer = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometmetriclayer.wmt2025esa.cn.result.jsonl'
xcomet = r'C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.cn.result.jsonl'


with open(xcometlayer, 'r', encoding='utf-8') as f:
    xcometmetriclayer = [json.loads(line)for line in f]

with open(xcomet, 'r', encoding='utf-8') as g:
    xcomet = [json.loads(line) for line in g]


def summary_stats_per_data(xcometmetriclayer, xcomet):

    print("Source Text", xcometmetriclayer[0]['src_text'])
    print("Target Text", xcometmetriclayer[0]['scores']['refA']['prediction'])

    print(xcometmetriclayer[0].keys())
    print(xcometmetriclayer[0]['scores']['refA'].keys())

    print("xcometlayer segment score:", xcometmetriclayer[0]['scores']['refA']['segment_score'])
    print("xcomet score:", xcomet[0]['scores']['refA']['system_score'])
    print('\n')
    print("xcometlayer error_span:", xcometmetriclayer[0]['scores']['refA']['error_span'])
    print('\n')
    print("xcomet error_span:", xcomet[0]['scores']['refA']['error_span'])
    print('\n')
    '''
    human annotations
    '''
    print(xcometmetriclayer[0]['scores']['refA']['human_score'])


if __name__=="__main__":
    summary_stats_per_data(xcometmetriclayer, xcomet)