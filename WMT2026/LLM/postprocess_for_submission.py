import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_priority', help='directory to files needed for postprocessing')
    parser.add_argument('--input_dir_nonpriority', help='directory to nonpriority files needed for postprocessing')
    parser.add_argument('--output', help='path to all concatenated files for submission')
    return parser.parse_args()


def get_all_jsonl(input_dir_priority, input_dir_nonpriority):
    '''
    Loads all jsonl files from each directory and collects every line into a list.
    '''
    data = []
    for directory in (input_dir_priority, input_dir_nonpriority):
        for filepath in sorted(glob.glob(os.path.join(directory, '*.jsonl'))):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # skip blank lines
                        data.append(json.loads(line))
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
        unwanted_fields_in_errors = ['text', 'confidence', 'entropy', 'category', 'content']
        for system in line['task1_pred']:
            if line['task1_pred'][system]['errors']:
                for error_span in line['task1_pred'][system]['errors']:
                    for field in unwanted_fields_in_errors:
                        if field in error_span:
                            error_span.pop(field)
                    #assert sorted(list(error_span.keys())) == sorted(['start', 'end', 'severity']), \
                    #f"system {system!r}: unexpected keys {sorted(error_span.keys())}"  
        return line

    def clamp_spans_to_length(line):
        for system in line['task1_pred']:
            sysdict = line['task1_pred'][system]
            pred = sysdict.get('prediction')
            if pred is None:
                continue                      # prediction already gone; can't clamp
            tgt_len = len(pred)
            spans = sysdict.get('refined_errors', sysdict.get('errors')) or []
            fixed = []
            for span in spans:
                start, end = span['start'], span['end']
                # clamp into [0, tgt_len]
                start = max(0, min(start, tgt_len))
                end   = max(0, min(end,   tgt_len))
                if start < end:               # keep only spans that still have width
                    span['start'], span['end'] = start, end
                    fixed.append(span)
            sysdict['refined_errors'] = fixed
        return line

    def fix_severity(line):
        allowed = {'major', 'minor'}
        for system in line['task1_pred']:
            sysdict = line['task1_pred'][system]
            spans = sysdict.get('refined_errors', sysdict.get('errors')) or []
            cleaned = []
            for span in spans:
                sev = span.get('severity')
                sev = sev.strip().lower() if isinstance(sev, str) else ''
                if sev not in allowed:
                    # decide fallback — see note below
                    sev = 'minor'
                span['severity'] = sev
                cleaned.append(span)
            sysdict['refined_errors'] = cleaned
        return line

    with open(output, 'w', encoding='utf-8') as f:
        for line in data:
            line = clamp_spans_to_length(line) 
            line = fix_severity(line)
            line = remove_unwanted_fields_per_system(line)
            line = remove_unwanted_fields_in_error(line)
            line.pop('task_pred', None)
            
            f.write(json.dumps(line, ensure_ascii=False ) + '\n')
        


def main():
    args = parse_args()
    data = get_all_jsonl(args.input_dir_priority, args.input_dir_nonpriority)
    postprocess(data, args.output)
    print("done!")


if __name__ == "__main__":
    main()

