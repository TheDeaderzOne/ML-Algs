from typing import List
from collections import defaultdict, Counter
from random import choices
from warnings import warn


class NGramLanguage:
    def __init__(self, N=3):
        self.N = max(2, int(N))
        self.NWords = {}
        self.NMinusWords = {}
        self.backup = {}

    def fit(self, trainsentences: List[List[str]], backupwords=None):
        """Make sure that you pad out the strings of length (N) beforehand with start (<s>) and stop (</s>) symbols, you should probably make the sentences all lowercase"""
        NList = []

        for sent in trainsentences:
            if len(sent) >= self.N:
                for i in range(len(sent) - self.N + 1):
                    NList.append((tuple(sent[i : i + self.N - 1]), sent[i + self.N - 1]))

        self.NMinusWords = Counter()

        self.NWords = defaultdict(Counter)

        self.backup = Counter(backupwords)

        for prefix, next_word in NList:
            self.NWords[prefix][next_word] += 1
            self.NMinusWords[prefix] += 1

    def predict_next_word(self, sentence: List[str]) -> str:
        """Should be a sentence (list) of preferably lowercase strings"""
        if len(sentence) < self.N - 1:
            key = tuple(["<s>"] * (self.N - 1 - len(sentence)) + sentence)
        else:
            key = tuple(sentence[-(self.N - 1) :])
        items = list(self.NWords[key].keys())
        weights = list(self.NWords[key].values())

        if self.backup is not None and len(items) == 0:
            # warn("Novel Pattern, imputed with randomly selected word")
            return choices(list(self.backup.keys()), weights=list(self.backup.values()), k=1)[0]

        return choices(items, weights=weights, k=1)[0]

    def generate_sentence(self, sent: List[str], iterations: int) -> List[str]:
        for _ in range(iterations):
            current_word = self.predict_next_word(sent)
            if current_word == "</s>":
                break
            else:
                sent.append(current_word)
        return sent

    # Implement Sentence Likelihood Later
    def sentence_likelhood(self, sentence: List[str]) -> float:
        return 0.0
