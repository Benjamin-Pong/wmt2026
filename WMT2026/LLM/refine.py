
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import torch

'''
LLMJudge QE metric that takes the outputs of xcomet and improves/refines the annotations
judge accounts for wmt2025 and wmt2026 results output
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", help='wmt2025 or wmt2026(exp)')
    parser.add_argument("--input", help='path to input file, which is the output of xcomet')
    parser.add_argument("--prompt", help='path to refined prompt')
    parser.add_argument("--refined_results", help="output path to the refined results")
    parser.add_argument("--model", help='path to huggingface model')
    return parser.parse_args()


def get_data(input):
    with open(input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def load_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.readlines()
        #print("prompt length",len(prompt))
    return "\n".join(prompt)

def inference(model, tokenizer, prompt) -> List[Dict]:
    messages = [
    {"role": "user", "content": f"{prompt}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=100000)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    
def refine_26(model, tokenizer, data,prompt,k, refined_results):
    '''
    checks/edge cases:
    1. sortedness
    2. top-k logic: what if the list has no errors, what if the list has less than k members?
    2. all fields are extracted correctly


    Outputs final set of data for evaluation (2026 specific format only!)
    do I want this or do I want some error analyses??
    '''
    with open(refined_results, 'w', encoding='utf-8') as f:

        for line in data:
            #extract all the required fields for a sample to be injected into a prompt
            #per system per line
            src_lang = line['item_id'].split("###")[1].split('_')[1]
            tgt_lang = line['item_id'].split('###')[2].split('_')[1]
            
            system_scores = line["task1_pred"]
            for system in system_scores:
                prediction = system_scores[system]['prediction']
                src_text = system_scores[system]['src']
                error_spans = system_scores[system]['errors']
                print(error_spans)
                if error_spans:
                    ranked_error_spans = sorted(error_spans, key=lambda d:d['entropy'], reverse=True)
                

                    top_k_spans = ranked_error_spans[:k] #these spans will be shown to llm prompt
                    curr_prompt = prompt.format(source_language=src_lang, target_language=tgt_lang, source_segment=src_text, target_segment=prediction, xcomet_error_spans=top_k_spans)
                    #print(curr_prompt)
                    refined_spans = inference(model, tokenizer, curr_prompt) #list of refined error span
                    print(refined_spans)

                    #refined_spans is a subset of ranked_error_spans
                    #remove ranked_error_spans 

                    system_scores[system]['refined_errors']=refined_spans
                    print(system_scores[system].keys())
            line['task_pred'] = system_scores       

            #f.write(json.dumps(line, ensure_ascii=False)+'\n')

                
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,   # half precision — ~64GB for 32B, fits your 80GB
    device_map="auto")
    
    data = get_data(args.input)
    prompt = load_prompt(args.prompt)
    if args.cycle=="wmt2026":
        refine_26(model, tokenizer, data[0:10],prompt, 3, args.refined_results)
    

if __name__ == "__main__":
    main()