import json
import argparse

'''
The test file is actually gigantic
1. Split up all the large file into language specific files in directory "mt2026_split"

2. Sort them in terms of priority, and run inference on each set  

Next step is to run inference on each set!
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument('--all_pairs')
    #parser.add_argument('--priority')
    
    return parser.parse_args()



def get_unique_langpairs(data, all_pairs):
    langpairs=set()
    for line in data:
        #print(line['item_id'])
        src_lang = line['item_id'].split("###")[1].split('_')[1]
        #print(src_lang)
        tgt_lang = line['item_id'].split('###')[2].split('_')[1]
        #print(tgt_lang)
        langpair = (src_lang, tgt_lang)
        
        langpairs.add(langpair)
    with open(all_pairs, 'w', encoding='utf-8') as f:
        for langpair in sorted(langpairs):
            f.write(str(langpair) + '\n')
    return langpairs

def split(data, langpairs):
    handles = {}
    total_lines = 0
    for line in data:
        src_lang = line['item_id'].split("###")[1].split('_')[1]
        tgt_lang = line['item_id'].split('###')[2].split('_')[1]
        if (src_lang, tgt_lang) not in handles:
            curr_lp_writer = open(f"mt2026_split/{src_lang}_{tgt_lang}.jsonl", 'w', encoding='utf-8')
            handles[(src_lang,tgt_lang)]=curr_lp_writer
        curr_lp_writer.write(json.dumps(line, ensure_ascii=False) + '\n')
        total_lines +=1

    assert total_lines == len(data)

def check_lines(data):
    import glob
    import os

    OUTPUT_DIR = "mt2026_split"
    original_lines = len(data)
    total_lines = 0
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl")):
        with open(path, 'r', encoding='utf-8') as f:
            split = [json.loads(line)for line in f]
            total_lines += len(split)
    print(original_lines)
    print(total_lines)
    assert original_lines == total_lines

        
    
        
   

args = parse_args()

with open(args.input, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]


langpairs = get_unique_langpairs(data, args.all_pairs)
split(data, langpairs)
check_lines(data)
