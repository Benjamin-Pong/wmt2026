import json
import argparse
'''
This script converts the output of llm refinement, such that it has all the required fields
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json')
    parser.add_argument('--output_json')
    return parser.parse_args()

def process(data, output):
    with open(output, 'w', encoding='utf-8') as f:
        for line in data:
            for system in line['scores']:
                s= line['scores'][system]
                print(s.keys())
                if 'refined_errors' in line['scores'][system]:
                 
                    s['error_span'] = s.pop('refined_errors')
                    #print(s['error_span'].keys())
                    
                s.pop('subword_probs')
                s.pop('logits')
                

            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    

def get_input_data(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def main():
    args = parse_args()
    input_data = get_input_data(args.input_json)
    process(input_data, args.output_json)

if __name__ == "__main__":
    main()


