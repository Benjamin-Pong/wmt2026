import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import os
import torch

'''
LLM metric using Gemba MQM Prompt,
input works only for WMT 2025 Data at the moment.
output is in the json format, identical to the output of xcomet script
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to original wmt file')
    parser.add_argument('--output', help='predictions by llm in the same format as xcomet')
    parser.add_argument('--model', help='llm model name in huggingface; deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')
    return parser.parse_args()
def get_systems_evaluated(line):
    return set(line['scores'].keys())

def inference(target_language, source_language, target_segment, source_segment) -> List[Dict]:
    prompt = '''You are an annotator for the quality of machine translation. Your task is to identify
        errors and assess the quality of the translation.
        (user) {source_language} source:\n
        ```{source_segment}```\n
        {target_language} machine translation:\n
        ```{target_segment}```\n
        \n
        Based on the source segment and machine translation surrounded with triple backticks, identify
        error types in the translation and classify them. The categories of errors are: accuracy
        (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar,
        inconsistency, punctuation, register, spelling),
        locale convention (currency, date, name, telephone, or time format)
        style (awkward), terminology (inappropriate for context, inconsistent use), non-translation,
        other, or no-error.\n

        Each error is classified as one of three categories: critical, major, and minor.
        Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what
        the text is trying to say is still understandable. Minor errors are technically errors,
        but do not disrupt the flow or hinder comprehension. \n

        Make sure to only evaluate the machine translation's error in {target_language}, given the source segment in {source_language}. Produce a JSON array of objects. Each object represents a single error and
        must have exactly these keys: "start", "end", "severity". start' and 'end' are character indices of the identified error span from the target language translation, and "severity" is one of
        "critical", "major", or "minor". If there are no errors, return an empty array [].
        Output only the JSON array, with no other text. 

        
        '''
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
    

def reconstruct(data, output):
    with open(args.output, 'w', encoding='utf-8') as g:
        for i, line in enumerate(data):
            human_scores = line['scores']
            src_lang = line['doc_id'].split('_')[0].split('-')[0]
            tgt_lang = line['doc_id'].split('_')[0].split('-')[1]
            domain = line['doc_id'].split('#')[1].strip('_')
            doc_id = line['doc_id']
            '''
            get relevant fields.
            '''

            src_text = line['src_text']
            evaluated_systems = get_systems_evaluated(line)
            relevant_text = {k:v for k,v in line['tgt_text'].items() if k in evaluated_systems}
            
            scores = {}
            for system in relevant_text:
                human_score_per_system = human_scores[system]
                prediction = relevant_text[system]
                print(f"source: {src_text}, target: {relevant_text[system]}")
                llm_error_spans = inference(tgt_lang, src_lang, relevant_text[system], src_text)
                scores[system]={'system_id': system, 'prediction':prediction, 'error_span':llm_error_spans, 'human_score':human_score_per_system}
                

            res_per_line = {'doc_id': doc_id, 'source_segment': src_text, 'source_lang': src_lang, 'target_lang': tgt_lang, 'domain': domain, 'scores': scores}
            g.write(json.dumps(res_per_line, ensure_ascii=False)+'\n')
            g.flush()
            os.fsync(f.fileno())
            print(f"line {i} written, file now {os.path.getsize(output):,} bytes", flush=True)
            



if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,   # half precision — ~64GB for 32B, fits your 80GB
    device_map="auto")

        
    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    reconstruct(data[0:10], args.output)
    
