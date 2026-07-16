from difflib import SequenceMatcher

def diff_spans(mt, pert):
    sm = SequenceMatcher(None, mt, pert, autojunk=False)
    return [(j1, j2) for tag, _, _, j1, j2 in sm.get_opcodes() if tag != "equal"]


a = 'there is a boy'
b = 'there is a bo'

print(diff_spans(b,a))


d={'src': 'hello', 'annotations':[{'start': '1', 'end':'2'}]}
print(type(d['annotations'][0]['start']))
for error_span in d['annotations']:
    for field in ('start', 'end'):
        error_span['start'] = int(error_span['start'])

print(type(d['annotations'][0]['start']))

