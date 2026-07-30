import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='directory to files needed for postprocessing')
    parser.add_argument('--output')
    return parser.parse_args()


def concat_files():
    '''
    Concatenates all the distributed-ly processed jsonl files
    '''

    pass

def unify_error_spans():
    pass

def postprocess(data):
    '''
    Postprocesses each jsonl line to remove all the unwanted fields for submission
    '''
    unwanted_fields_in_system = ['avg_confidence', 'avg_entropy','logits', 'subword_probs']
    unwanted_fields_in_errors = ['text', 'confidence', 'entropy']
    def remove_unwanted_fields_per_system(unwanted_fields_in_system, line):
        for field in unwanted_fields_in_system: 
            for system in line['task1_pred']:
                line['task1_pred'][system].pop(field)
        return line

    def remove_unwanted_fields_in_error(unwanted_fields_in_errors, line):
        
        for system in line['task1_pred']:
            if line['task1_pred'][system]['errors']:
                for error_span in line['task1_pred'][system]['errors']:
                    for field in unwanted_fields_in_errors:
                        error_span.pop(field)

            line['task1_pred'][system]['errors'] = unify_error_spans(line['task1_pred'][system]['errors'], line['task1_pred'][system]['refined_errors'])
                
        return line

    def unify_error_spans(errors, refined_errors):
        '''
        this function unifies original error spans and the refined error spans
        '''
        fo
        


    for line in data:
        
        line = remove_unwanted_fields_per_system(unwanted_fields_in_system, line)
        line = remove_unwanted_fields_in_error(unwanted_fields_in_errors, line)
        line = unify_error_spans(line)


    pass

def main():
    args = parse_args()
    

