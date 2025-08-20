import numpy as np
from collections import Counter
import sys
import os
from nltk.corpus import ieer
from nltk import tree2conlltags
from nltk import Tree


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from CompletedMLAlgorithms.HiddenMarkovModel import HMM

# print(train_data.features)
# print(train_data.features['ner_tags'].feature.names)

all_sents = []


for doc in ieer.parsed_docs():
    for sent in doc.text:
        if isinstance(sent, Tree):
            try:
                iob = tree2conlltags(sent)
                # keep only (word, ner) instead of (word, pos, ner)
                simplified = [(w, ner) for (w, pos, ner) in iob]
                all_sents.append(simplified)
            except Exception as e:
                # skip any malformed sentence
                # (rare edge cases: empty children, bad formatting)
                continue


print(all_sents)


class NER_HMM(HMM):
    def __init__(self, data):

        pass
