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

def postprocess(data):
    '''
    Postprocesses each jsonl line to remove all the unwanted fields for submission
    '''
    
    def remove_unwanted_fields_per_system(line):
        unwanted_fields_in_system = ['avg_confidence', 'avg_entropy','logits', 'subword_probs', 'reasoing_trace']
        for field in unwanted_fields_in_system: 
            for system in line['task1_pred']:
                if field in line['task1_pred'][system]:
                    line['task1_pred'][system].pop(field)
                line['task1_pred'][system]['old_errors'] = line['task1_pred'][system].pop('errors')
                line['task1_pred'][system]['errors'] = line['task1_pred'][system].pop('refined_errors')
                line['task1_pred'][system].pop('old_errors')
                line['task1_pred'][system]['omission']= None

                assert sorted(list(line['task1_pred'][system].keys())) == sorted(['errors', 'omission'])
        return line



        return line

    def remove_unwanted_fields_in_error(line):
        unwanted_fields_in_errors = ['text', 'confidence', 'entropy']
        for system in line['task1_pred']:
            if line['task1_pred'][system]['errors']:
                for error_span in line['task1_pred'][system]['errors']:
                    for field in unwanted_fields_in_errors:
                        if field in error_span:
                            error_span.pop(field)

                    assert sorted(list(error_span.keys())) == sorted(['start', 'end', 'severity'])
                
        return line

    def rename_and_remove(line):
        '''
        rename refined errors to old_errors
        rename refined errors to errors
        remove old errors entry
        '''
        
        

        
    for line in data:
        
        line = remove_unwanted_fields_per_system(unwanted_fields_in_system, line)
        line = remove_unwanted_fields_in_error(unwanted_fields_in_errors, line)
        line = unify_error_spans(line)


    pass

def main():
    args = parse_args()
    

