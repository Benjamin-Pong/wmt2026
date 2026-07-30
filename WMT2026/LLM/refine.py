
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import torch
import re
import copy

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
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

def parse_response(raw_output):
    '''
    this function parses llm response to extract only the json array
    '''

    text = raw_output

    # 1. Drop the reasoning block — everything before and including </think>.
    #    This prevents grabbing a stray [...] that appears inside the thinking.
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 2. Strip the model's end-of-sequence token if present.
    text = text.replace("<｜end▁of▁sentence｜>", "").strip()

    # 3. Prefer a fenced ```json ... ``` block if there is one.
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # 4. Otherwise grab the last top-level [...] in the remaining text.
        matches = re.findall(r"\[.*?\]", text, re.DOTALL)
        candidate = matches[-1] if matches else None

    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
def remove_unwanted_fields_in_error(errors):
    unwanted_fields_in_errors = ['text', 'confidence', 'entropy']
    for error_span in errors:
        for field in unwanted_fields_in_errors:
            error_span.pop(field)
    return errors

def refine_25(model, tokenizer, data,prompt,k, refined_results):
    with open(refined_results, 'w', encoding='utf-8') as f:
        for line in data:
            #extract all the required fields for a sample to be injected into a prompt
            #per system per line
            src_lang = line['doc_id'].split('_')[0].split('-')[0]
            tgt_lang = line['doc_id'].split('_')[0].split('-')[1]
            src_text = line['source_segment']
            
            system_scores = line["scores"]
            for system in system_scores:
                prediction = system_scores[system]['prediction']
                error_spans = system_scores[system]['error_span']
                
                ranked_error_spans = sorted(error_spans, key=lambda d:d['entropy'], reverse=True)
                top_k_spans = ranked_error_spans[:k+1] #these spans will be shown to llm prompt
                unchanged_spans = ranked_error_spans[k+1:] #for unification later
                curr_prompt = prompt.format(source_language=src_lang, target_language=tgt_lang, source_segment=src_text, target_segment=prediction, xcomet_error_spans=top_k_spans)
        
                corrected_topk_spans = inference(model, tokenizer, curr_prompt) 
                system_scores[system]['reasoning_trace']=corrected_topk_spans
                print("reasoning trace", corrected_topk_spans)
                corrected_topk_spans = parse_response(corrected_topk_spans) #list of dicts
                print("parsed", corrected_topk_spans) 
                system_scores[system]['corrected_topk_spans']=corrected_topk_spans

                cleaned_corrected_topk_spans = remove_unwanted_fields_in_error(corrected_topk_spans)
                cleaned_unchanged_spans = remove_unwanted_fields_in_error(unchanged_spans)

                system_scores[system]['refined_errors'] = cleaned_unchanged_spans.extend(cleaned_corrected_topk_spans)
                print(system_scores[system]['refined_errors'])
                print(system_scores[system].keys())
            line['task_pred'] = system_scores       

            f.write(json.dumps(line, ensure_ascii=False)+'\n')



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
                #print(error_spans)
         
                ranked_error_spans = sorted(error_spans, key=lambda d:d['entropy'], reverse=True)
                top_k_spans = ranked_error_spans[:k+1] #these spans will be shown to llm prompt
                unchanged_spans = ranked_error_spans[k+1:] #for unification later
                curr_prompt = prompt.format(source_language=src_lang, target_language=tgt_lang, source_segment=src_text, target_segment=prediction, xcomet_error_spans=top_k_spans)
        
                corrected_topk_spans = inference(model, tokenizer, curr_prompt) 
                system_scores[system]['reasoning_trace']=corrected_topk_spans
                print("reasoning trace", corrected_topk_spans)
                corrected_topk_spans = parse_response(corrected_topk_spans) #list of dicts
                print("parsed", corrected_topk_spans) 
                system_scores[system]['corrected_topk_spans']=corrected_topk_spans

                cleaned_corrected_topk_spans = remove_unwanted_fields_in_error(corrected_topk_spans)
                cleaned_unchanged_spans = remove_unwanted_fields_in_error(unchanged_spans)

                

                system_scores[system]['refined_errors'] = cleaned_unchanged_spans.extend(cleaned_corrected_topk_spans)
                print(system_scores[system]['refined_errors'])

                

                print(system_scores[system].keys())
            line['task_pred'] = system_scores       

            f.write(json.dumps(line, ensure_ascii=False)+'\n')

                
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,   # half precision — ~64GB for 32B, fits your 80GB
    device_map="auto")
    
    data = get_data(args.input)
    prompt = load_prompt(args.prompt)
    if args.cycle=="wmt2026":
        refine_26(model, tokenizer, data[0:10],prompt, 5, args.refined_results)
    

if __name__ == "__main__":
    main()