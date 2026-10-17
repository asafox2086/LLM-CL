"""Primary Answer F1: exactly the existing stored token_f1, no option mapping."""
import re
from collections import Counter

VERSION = 'answer-f1-existing-token-v1-20261017'


def tokens(text):
    return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', text.lower())


def answer_f1(prediction, references):
    assert isinstance(prediction,str) and isinstance(references,list) and references
    a=tokens(prediction);values=[]
    for reference in references:
        b=tokens(reference)
        if not a or not b:
            values.append(100.*float(a==b))
        else:
            overlap=sum((Counter(a)&Counter(b)).values())
            values.append(100*(2*overlap/(len(a)+len(b))))
    return max(values)
