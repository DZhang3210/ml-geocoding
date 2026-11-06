import re


def tokenize(s):
    # the tokenisation word2vec was trained with; ESIM must use the same one
    return re.findall(r"[A-Za-z]+|\d+", s.upper())
