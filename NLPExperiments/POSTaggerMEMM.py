import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import random


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from NLPExperiments.POSTaggerHMM import POS_HMM


def list_accuracy_tuple(list1, list2):
    return sum(x == y for x, y in zip(list1, list2)), len(list1)


class POS_MEMM(POS_HMM):

    def _tags(self, sentences):
        all_tags = set()
        for sentence in sentences:
            for word, tag in sentence:
                all_tags.add(tag)
        tag_list = sorted(list(all_tags))
        return tag_list

    def _default_filter(self, word):
        word_shape = "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in word)
        new_dict = {
            f"word={word}": 1,
            "capitalization": int(word.isupper()),
            f"word_shape={word_shape}": 1,
            "has_hyphen": int("-" in word),
            "has_apostrophe": int("'" in word),
            "word_length": min(len(word), 10),
            "has_digit": int(any(c.isdigit() for c in word)),
            "is_all_caps": int(word.isupper()),
            "is_title_case": int(word.istitle()),
        }
        for i in range(1, min(4, len(word) + 1)):
            new_dict[f"prefix_{i}={word[:i].lower()}"] = 1
            new_dict[f"suffix_{i}={word[-i:].lower()}"] = 1
        return new_dict

    def __init__(self, corpus, train_test_split=1.0):

        sents = list(corpus.tagged_sents())
        random.seed(43)
        random.shuffle(sents)
        train_size = int((len(sents) * train_test_split))
        self.testsents = sents[train_size:]
        sents = sents[:train_size]

        self.tag_list = self._tags(sents)
        list_of_dicts = []
        y_labs = []

        for sentence in sents:
            prev_word = None
            prev_tag = "START"
            for word, tag in sentence:
                feature_dict = self._default_filter(word)
                feature_dict[f"prev_tag={prev_tag}"] = 1
                if prev_word is not None:
                    feature_dict[f"prev_word={prev_word}"] = 1
                prev_word = word
                prev_tag = tag
                y_labs.append(tag)
                list_of_dicts.append(feature_dict)

        self.vectorizer = DictVectorizer(sparse=True)
        feature_matrix = self.vectorizer.fit_transform(list_of_dicts)
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000, verbose=1)
        # self.model = LogisticRegression(solver="liblinear", max_iter=5000, verbose=1)
        self.model.fit(feature_matrix, y_labs)
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.model.classes_)}

    def decode(self, observations):

        new_arr = np.zeros((len(self.tag_list), len(observations)))
        s_arr = np.zeros((len(self.tag_list), len(observations)), dtype=int)

        # Create the feature dictionary once (since it's the same for all tags)
        word = observations[0]
        feats = self._default_filter(word)
        feats["prev_tag=START"] = 1
        x = self.vectorizer.transform(feats)
        probs = self.model.predict_proba(x)[0]  # Shape: (num_classes,)

        # Now correctly map each tag to its probability
        for t in range(len(self.tag_list)):
            tag_name = self.tag_list[t]
            tag_idx = self.tag_to_idx[tag_name]
            new_arr[t, 0] = np.log(probs[tag_idx] + 1e-12)

        for y in range(1, len(observations)):
            feat_list = []
            for h in range(len(self.tag_list)):
                word = observations[y]
                feat = self._default_filter(word)
                feat[f"prev_word={observations[y-1]}"] = 1
                feat[f"prev_tag={self.tag_list[h]}"] = 1
                feat_list.append(feat)

            matr = self.vectorizer.transform(feat_list)
            probs = self.model.predict_proba(matr)

            for i in range(len(self.tag_list)):
                tag_idx = self.tag_to_idx[self.tag_list[i]]
                new_arr[i, y] = np.max(new_arr[:, y - 1] + np.log(probs[:, tag_idx] + 1e-12))
                s_arr[i, y] = int(np.argmax(new_arr[:, y - 1] + np.log(probs[:, tag_idx] + 1e-12)))

        score = np.exp(np.max(new_arr[:, -1]))

        best_row_pointer = np.argmax(new_arr[:, -1])
        best_row = [0] * len(observations)

        for j in range(len(observations) - 1, -1, -1):
            best_row[j] = int(best_row_pointer)
            if best_row_pointer >= 0:
                best_row_pointer = s_arr[best_row_pointer][j]

        hidden_states = [self.tag_list[x] for x in best_row]

        return hidden_states, score
