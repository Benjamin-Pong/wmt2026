from pathlib import Path
from dotenv import load_dotenv
from comet import download_model, load_from_checkpoint
from comet.models.multitask.xcomet_layerwise_metric import XCOMETMetricLayer
from comet.models.multitask.xcomet_continuous_metric import XCOMETContinuousMetric
import os
from huggingface_hub import whoami
import json
import argparse
import inspect


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


print("Token loaded:", os.environ.get("HF_TOKEN") is not None)
print(whoami())


#print("source text", human_eval[0]['src_text'])
#print('all predictions', len(human_eval[1]['tgt_text']))
#print(len(human_eval))
#example = human_eval[0]['tgt_text'] #de
#print(example)
#print("dictionary keys for 1 item:" ,human_eval[0].keys())
#print('systems evaluated', set(human_eval[0]['scores'].keys()))

#does

def parse_args():
    parser = argparse.ArgumentParser(description='xcomet inference using different model scripts')
    parser.add_argument("--model", type=str, required=False, help='Name of custom xcomet callibration model')
    parser.add_argument("--input_file", type=str, required=True, help='path to input file for xcomet inference')
    parser.add_argument("--output_file", type=str, required=True, help='path to inference results')
    parser.add_argument("--locale", type=str, required=False, help='initialize specific locale or ALL if running inference on the full corpus.' )
    parser.add_argument("--mode", type=str, required=False, help= 'either write or append to output json file')

    return parser.parse_args()



def get_systems_evaluated(line):
    return set(line['scores'].keys())

def score(human_eval, xcomet, output_file, locale, mode):
    '''
    Function scores predictions with batching
    xcomet scores all systems per line at one shot
    '''
    with open(output_file, mode, encoding='utf-8') as f:
        for i, line in enumerate(human_eval):
            human_scores = line['scores']
            src_lang = line['doc_id'].split('_')[0].split('-')[0]
            tgt_lang = line['doc_id'].split('_')[0].split('-')[1]
            domain = line['doc_id'].split('#')[1].strip('_')
            doc_id = line['doc_id']
            if locale!='ALL':
                if src_lang == 'en' and tgt_lang==locale:
                    src_text = line['src_text']
                    evaluated_systems = get_systems_evaluated(line)
                    relevant_text = {k:v for k,v in line['tgt_text'].items() if k in evaluated_systems}
                    
            
                    scores = {}
                    for system in relevant_text:
                        human_score_per_system = human_scores[system]
                        prediction = relevant_text[system]
                        data = [
                            {
                                "src": f"{src_text}",
                                "mt": f"{prediction}",
                            
                            }
                        ]
                        model_output = xcomet.predict(data, batch_size=128, gpus=1)
                        # Segment-level scores
                        #print (model_output.scores)
                
                        # System-level score
                        #print (model_output.system_score) 
                
                        #print (model_output.metadata.error_spans)
                
                        scores[system]={'doc_id': line['doc_id'], 'system_id': system, 'prediction':prediction,'segment_score':model_output.scores[0], 'system_score':model_output.system_score, 'error_span':model_output.metadata.error_spans[0], 'logits': model_output.metadata.logits, 'subword_probs': model_output.metadata.subword_probs, 'human_score':human_score_per_system}
            
                    res_per_line = {'source_segment': src_text, 'source_lang': src_lang, 'target_lang': tgt_lang, 'scores': scores}
                    f.write(json.dumps(res_per_line, ensure_ascii=False))
                    f.write('\n')
            else:
                src_text = line['src_text']
                evaluated_systems = get_systems_evaluated(line)
                relevant_text = {k:v for k,v in line['tgt_text'].items() if k in evaluated_systems}
                data = [] #collects src-tgt text pairs per system
                
        
                scores = {}
                for system in relevant_text:
                    prediction = relevant_text[system]
                    data.append(
                        {
                            "src": f"{src_text}",
                            "mt": f"{prediction}",
                        
                        }
                    )
                model_output = xcomet.predict(data, batch_size=32, gpus=1)
                segment_scores = model_output.scores
                #system_scores = model_output.system_scores
                error_spans = model_output.metadata.error_spans
                logits = model_output.metadata.logits
                subword_probs = model_output.metadata.subword_probs
                tokens = model_output.metadata.tokens
                token_ids = model_output.metadata.token_ids

                for idx, system in enumerate(relevant_text):
                    human_score_per_system = human_scores[system]
                    prediction = relevant_text[system]
                    scores[system]={'system_id': system, 'prediction':prediction,'segment_score':segment_scores[idx], 'error_span':error_spans[idx], 'logits': logits[idx], 'subword_probs': subword_probs[idx], 'tokens': tokens[idx], 'token_ids': token_ids[idx], 'human_score':human_score_per_system}
        
                res_per_line = {'doc_id': doc_id, 'source_segment': src_text, 'source_lang': src_lang, 'target_lang': tgt_lang, 'domain': domain, 'scores': scores}
                f.write(json.dumps(res_per_line, ensure_ascii=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
                print(f"line {i} written, file now {os.path.getsize(output_file):,} bytes", flush=True)
                

if __name__ == "__main__":
    args = parse_args()
    model_path = download_model("Unbabel/XCOMET-XL")
    if args.model:
        model = globals()[args.model] #maps model's string name to model object
        print(f"model experiment: {model}")
        xcomet = model.load_from_checkpoint(model_path, strict=False)
        print(inspect.getfile(type(xcomet))) #prints path to custom xcomet

    else:
        xcomet = load_from_checkpoint(model_path)

    human_eval_file=args.input_file
    output_file = args.output_file
 
    with open(human_eval_file, 'r', encoding='utf-8') as f:
      human_eval = [json.loads(line) for line in f]

    if args.mode == 'a':
        print("append mode slice output file")
        with open(output_file, 'r', encoding='utf-8') as g:
            out = [json.loads(line) for line in g]
            last_index = len(out)-1
            human_eval_start_idx = last_index+1
            human_eval = human_eval[human_eval_start_idx:]
            print(f'continuing from line {human_eval_start_idx}')
    score(human_eval, xcomet, output_file,args.locale, args.mode)
    

  