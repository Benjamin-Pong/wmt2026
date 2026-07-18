import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

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

def get_systems_evaluated(line):
    return set(line['scores'].keys())

def predict():
    pass

def inference(target_language, source_language, target_segment, source_segment) -> List[Dict]:
    prompt = '''You are an annotator for the quality of machine translation. Your task is to identify
        errors and assess the quality of the translation.
        (user) {source_language} source:\n
        ```{source_segment}```\n
        {target_language} translation:\n
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
        but do not disrupt the flow or hinder comprehension.'''
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

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    pass



def reconstruct(data):
    with open(args.output, 'w', encoding='utf-8') as g:
        for line in data:
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
                prediction = relevant_text[system]
                llm_error_spans = inference(tgt_lang, src_lang, relevant_text[system], src_text)
                scores[system]={'system_id': system, 'prediction':prediction,'segment_score':segment_scores[idx], 'error_span':error_spans[idx], 'logits': logits[idx], 'subword_probs': subword_probs[idx], 'tokens': tokens[idx], 'token_ids': token_ids[idx], 'human_score':human_score_per_system}
                

            res_per_line = {'doc_id': doc_id, 'source_segment': src_text, 'source_lang': src_lang, 'target_lang': tgt_lang, 'domain': domain, 'scores': scores}
            f.write(json.dumps(res_per_line, ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
            print(f"line {i} written, file now {os.path.getsize(output_file):,} bytes", flush=True)
            


            
            g.write(json.dumps(response, ensure_ascii=False) + '\n')




if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

        
    

    with open(args.input, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    reconstruct
    
