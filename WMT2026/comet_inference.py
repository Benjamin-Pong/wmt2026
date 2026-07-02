from pathlib import Path
from dotenv import load_dotenv
from comet import download_model, load_from_checkpoint
from comet.models.multitask.xcomet_layerwise_metric import XCOMETMetricLayer
from comet.models.multitask.xcomet_continuous_metric import XCOMETContinuousMetric
import os
from huggingface_hub import whoami
import json
import argparse

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
  
    return parser.parse_args()



def get_systems_evaluated(line):
    return set(line['scores'].keys())

def score(human_eval, xcomet, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in human_eval:
            human_scores = line['scores']
            src_lang = line['doc_id'].split('_')[0].split('-')[0]
            tgt_lang = line['doc_id'].split('_')[1].split('-')[0]
            #if src_lang == 'en' and tgt_lang=='CN':
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
                model_output = xcomet.predict(data, batch_size=8, gpus=1)
                # Segment-level scores
                #print (model_output.scores)
        
                # System-level score
                #print (model_output.system_score) 
        
                #print (model_output.metadata.error_spans)
        
                scores[system]={'prediction':prediction,'segment_score':model_output.scores[0], 'system_score':model_output.system_score, 'error_span':model_output.metadata.error_spans[0], 'human_score':human_score_per_system}
    
            res_per_line = {'src_text': src_text, 'src_lang': src_lang, 'tgt_lang': tgt_lang, 'scores': scores}
            f.write(json.dumps(res_per_line, ensure_ascii=False))
            f.write('\n')

if __name__ == "__main__":
    args = parse_args()
    model_path = download_model("Unbabel/XCOMET-XL")
    if args.model:
        model = globals()[args.model]
        print(f"model experiment: {model}")
        xcomet = model.load_from_checkpoint(model_path, strict=False)
  
    xcomet = load_from_checkpoint(model_path)

    human_eval_file=args.input_file
    output_file = args.output_file
 
    with open(human_eval_file, 'r', encoding='utf-8') as f:
      human_eval = [json.loads(line) for line in f]

    score(human_eval, xcomet, output_file)
  