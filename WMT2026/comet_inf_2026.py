from pathlib import Path
from dotenv import load_dotenv
from comet import download_model, load_from_checkpoint
from comet.models.multitask.xcomet_layerwise_metric import XCOMETMetricLayer
from comet.models.multitask.xcomet_continuous_metric import XCOMETContinuousMetric
from comet.models.multitask.xcomet_logits_offset import XCOMETMetricLogitsAdj
import os
from huggingface_hub import whoami
import json
import argparse
import inspect
import copy


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
    parser.add_argument("--experiment_file", required=False, help="path to inference output file with additional fields")
    parser.add_argument("--submission_file", type=str, required=True, help='path to inference output file - only required fields by wmt2026')
    parser.add_argument("--mode", type=str, required=False, help= 'either write or append to output json file')

    return parser.parse_args()



def get_systems_evaluated(line):
    return set(line['hyps'].keys())

def score(human_eval, xcomet, submission_file, experiment_file, mode):
    '''
    Function scores predictions with batching
    xcomet scores all systems per line at one shot
    '''
    with open(submission_file, mode,  encoding='utf-8') as f , open(experiment_file, mode, encoding='utf-8') as g:
        empty_error = 0
        for i, line in enumerate(human_eval):

            #global variables to include in output
            item_id = line['item_id']
            src_text = line['src']
            #evaluated_systems = get_systems_evaluated(line)
            #relevant_text = {k:v for k,v in line['tgt_text'].items() if k in evaluated_systems}
            data = [] #collects src-tgt text pairs per system for batching of all system predictions

            all_predictions = line['hyps'] #keys are system names


    
            scores_exp = {}
            scores_submission={}
            for system in all_predictions:
                prediction = all_predictions[system]
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
            avg_conf = model_output.metadata.avg_confidence
            avg_ent = model_output.metadata.avg_entropy
        

            for idx, system in enumerate(all_predictions):
                #human_score_per_system = human_scores[system] NOT RELEVANT FOR TEST SET
                prediction = all_predictions[system]
                keys_remove = ['text', 'entropy', 'confidence']
                error_span_clean_sub = copy.deepcopy(error_spans[idx])
                error_span_clean_exp = copy.deepcopy(error_spans[idx])
                for key in keys_remove:
                    if error_span_clean_sub!=[]:
                        for error_span in error_span_clean_sub:
                            error_span.pop(key)
                            if error_span['severity']=='critical':
                                error_span['severity']= 'major'
                if error_span_clean_exp!=[]:
                    for error_span in error_span_clean_exp:
                        if error_span['severity']=='critical':
                            error_span['severity']= 'major'

                scores_exp[system]={'system_id': system, 'src': src_text,'prediction':prediction, 'avg_confidence':avg_conf[idx], 'avg_entropy': avg_ent[idx], 'errors':error_span_clean_exp, 'logits': logits[idx], 'subword_probs': subword_probs[idx]}

                scores_submission[system]={'errors': error_span_clean_sub, 'omission': None}
        
    
            res_sub_per_line = {'item_id': item_id, 'task1_pred': scores_submission}
            res_exp_per_line = {'item_id': item_id, 'task1_pred': scores_exp}

            f.write(json.dumps(res_sub_per_line, ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
            g.write(json.dumps(res_exp_per_line, ensure_ascii=False)+'\n')
            g.flush()
            os.fsync(g.fileno())
            print(f"line {i} written, file now {os.path.getsize(submission_file):,} bytes", flush=True)
            print(f"line {i} written, file now {os.path.getsize(experiment_file):,} bytes", flush=True)

            

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
    submission_file = args.submission_file
    experiment_file = args.experiment_file
 
    with open(human_eval_file, 'r', encoding='utf-8') as f:
      human_eval = [json.loads(line) for line in f]

    if args.mode == 'a':
        print("append mode slice output file")
        with open(submission_file, 'r', encoding='utf-8') as g:
            out = [json.loads(line) for line in g]
            last_index = len(out)-1
            human_eval_start_idx = last_index+1
            human_eval = human_eval[human_eval_start_idx:]
            print(f'continuing from line {human_eval_start_idx}')
    score(human_eval, xcomet, submission_file, experiment_file, args.mode)
    

  