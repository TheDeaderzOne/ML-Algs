import numpy as np
from typing import List
from collections import Counter

import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (the one containing 'CompletedMLAlgorithms')
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python search path
sys.path.append(parent_dir)

# Now your imports should work
from CompletedMLAlgorithms.HiddenMarkovModel import HMM

# remember the unknown tag, unk
# anything in the vocab only used twice or less gets turned to unknown
# we just need initial probabilities
# add-one smoothing


class POS_HMM(HMM):

    def _tags(self, sentences):
        all_tags = set()
        for sentence in sentences:
            for word, tag in sentence:
                all_tags.add(tag)
        tag_list = sorted(list(all_tags))
        tag_encoding = {item: index for index, item in enumerate(tag_list)}
        return tag_list, tag_encoding

    def _words(self, sentences):
        word_count = Counter()
        for s in sentences:
            for w in range(len(s)):
                word_count[s[w][0]] += 1
        word_encoding = {}
        i = 0
        for word, count in word_count.items():
            if count >= 2:
                word_encoding[word] = i
                i += 1
        word_encoding["<UNK>"] = i
        return word_encoding

    def _fit(self, sentences):
        for s in sentences:
            prev_tag = 0
            for w in range(len(s)):
                current_word = s[w][0]
                if s[w][0] not in self.encoding:
                    current_word = "<UNK>"
                word_ind = self.encoding[current_word]
                tag_ind = self.hidden_encoding_dict[s[w][1]]
                if w == 0:
                    self.init_prob_dict[s[w][1]] += 1
                else:
                    prev_tag_ind = self.hidden_encoding_dict[prev_tag]
                    self.markov_matrix[prev_tag_ind][tag_ind] += 1
                self.emission_matrix[tag_ind][word_ind] += 1
                prev_tag = s[w][1]

    def __init__(self, corpus):

        sents = corpus.tagged_sents()

        self.hidden_encoding, self.hidden_encoding_dict = self._tags(sents)
        self.encoding = self._words(sents)

        self.emission_matrix = np.zeros((len(self.hidden_encoding), len(self.encoding)))
        self.markov_matrix = np.zeros((len(self.hidden_encoding), len(self.hidden_encoding)))
        self.init_prob_dict = {key: 0 for key in self.hidden_encoding_dict}

        self._fit(sents)

        self.markov_matrix = self.markov_matrix / np.sum(self.markov_matrix, axis=1, keepdims=True)
        self.emission_matrix = self.emission_matrix / np.sum(self.emission_matrix, axis=1, keepdims=True)
        self.init_prob = np.array(list(self.init_prob_dict.values()), dtype=float)
        self.init_prob /= self.init_prob.sum()

    def _build_observ_matr(self, observations):
        temp_list = []
        for o in observations:
            if o in self.encoding:
                temp_list.append(self.emission_matrix[:, self.encoding[o]].reshape(-1, 1))
            else:
                temp_list.append(self.emission_matrix[:, self.encoding["<UNK>"]].reshape(-1, 1))

        return np.column_stack(temp_list)

    def likelihood(self, hidden_states, observations, init_prob=None) -> float:
        return super().likelihood(hidden_states, observations, self.init_prob)

    def decode(self, observations, initial_probabilities=None):
        return super().decode(observations, self.init_prob)
