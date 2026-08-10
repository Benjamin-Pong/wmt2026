import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_dir_priority', help='directory to files needed for postprocessing')
    parser.add_argument('--input_file', help='directory to nonpriority files needed for postprocessing')
    parser.add_argument('--output', help='path to all concatenated files for submission')
    return parser.parse_args()


def get_all_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def postprocess(data, output):
    '''
    Postprocesses each jsonl line to remove all the unwanted fields for submission
    '''
    
    def remove_unwanted_fields_per_system(line):
        unwanted_fields_in_system = ['avg_confidence', 'avg_entropy', 'logits',
                                    'subword_probs', 'reasoning_trace', 'corrected_topk_spans','prediction', 'src', 'system_id']
        for system in line['task1_pred']:
            sysdict = line['task1_pred'][system]
            for field in unwanted_fields_in_system:
                sysdict.pop(field, None)
            # rename refined_errors -> errors (dropping the original errors)
            sysdict['errors'] = sysdict.pop('refined_errors')
            sysdict['omission'] = None
            print(sysdict.keys())
            assert sorted(sysdict.keys()) == sorted(['errors', 'omission']), \
            f"system {system!r}: unexpected keys {sorted(sysdict.keys())}"
        return line


    def remove_unwanted_fields_in_error(line):
        unwanted_fields_in_errors = ['text', 'confidence', 'entropy', 'category']
        for system in line['task1_pred']:
            if line['task1_pred'][system]['errors']:
                for error_span in line['task1_pred'][system]['errors']:
                    for field in unwanted_fields_in_errors:
                        if field in error_span:
                            error_span.pop(field)
                    assert sorted(list(error_span.keys())) == sorted(['start', 'end', 'severity']), \
                    f"system {system!r}: unexpected keys {sorted(error_span.keys())}"  
        return line

    with open(output, 'w', encoding='utf-8') as f:
        for line in data:
            line = remove_unwanted_fields_per_system(line)
            line = remove_unwanted_fields_in_error(line)
            line.pop('task_pred')
            
            f.write(json.dumps(line, ensure_ascii=False ) + '\n')
        


def main():
    args = parse_args()
    data = get_all_jsonl(args.input_file)
    postprocess(data, args.output)
    print("done!")


if __name__ == "__main__":
    main()

