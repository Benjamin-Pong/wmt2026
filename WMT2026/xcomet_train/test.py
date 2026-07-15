from difflib import SequenceMatcher

def diff_spans(mt, pert):
    sm = SequenceMatcher(None, mt, pert, autojunk=False)
    return [(j1, j2) for tag, _, _, j1, j2 in sm.get_opcodes() if tag != "equal"]


a = 'there is a boy'
b = 'there is a bo'

print(diff_spans(b,a))
