import json
import argparse

'''
LLMJudge QE metric that takes the outputs of xcomet and improves/refines the annotations
judge accounts for wmt2025 and wmt2026 results output
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", help='wmt2025 or wmt2026(exp)')
    parser.add_arguments("--input", help='path to input file, which is the output of xcomet')
    parser.add_argument("--refined_results", help="output path to the refined results")
    return parser.parse_args()


def get_data(input):
    with open(input, 'r', encoding='utf-8') as f:
        data = [json.dumps(line) for line in f]
    return data





    return {'src_lang': src_lang}
def refine_26(data, k):
    '''
    checks/edge cases:
    1. sortedness
    2. top-k logic: what if the list has no errors, what if the list has less than k members?
    2. all fields are extracted correctly


    Outputs final set of data for evaluation (2026 specific format only!)
    do I want this or do I want some error analyses??
    '''
    for line in data:

        #extract all the required fields for a sample to be injected into a prompt
        #per system per line
        src_lang = line['item_id'].split("###")[1].split('_')[1]
        tgt_lang = line['item_id'].split('###')[2].split('_')[1]
        system_scores = line["task_pred"]
        for system in system_scores:
            prediction = system_scores['prediction']
            error_spans = system_scores['errors']
            if error_spans!=[]:
                ranked_error_spans = sorted(error_spans, key=lambda d:d['entropy'], reverse=True)
                sorted_error_spans = sorted(data, key=lambda d: d['x'], reverse=True)

                top_k_spans = sorted_error_spans[:k] #these spans will be shown to llm prompt

                refined_spans = inference(prompt, top_k_spans) #list of refined dictionaries

                #refined_spans is a subset of ranked_error_spans
                #remove ranked_error_spans 
                
            else:
                '''
                Else, make sure if it in the right format with all the desired keys
                '''

# [{'x': 3}, {'x': 2}, {'x': 1}]

       
        
        for system in system_scores:


    pass
def main():
    args = parse_args()
    data = get_data(args.input)
    if args.cycle=="wmt2026":
        refine_26()
    else:
        refine_25()

    pass

if __name__ == "__main__":
    main()