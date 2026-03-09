import numpy as np
from collections import Counter


class TextProcessor:
    def __init__(self, window_size=2, negative_samples=5):
        self.window_size = window_size
        self.neg_samples = negative_samples
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.unigram_table = None

    def build_vocab(self, sentences):
        words = [word for sent in sentences for word in sent.split()]
        counts = Counter(words)
        self.id2word = {i: word for i, (word, _) in enumerate(counts.items())}
        self.word2id = {word: i for i, word in self.id2word.items()}
        self.vocab_size = len(self.id2word)

        # Create unigram distribution for negative sampling (freq ^ 0.75).
        freqs = np.array([counts[self.id2word[i]] for i in range(self.vocab_size)], dtype=np.float64)
        pow_freqs = freqs**0.75
        self.unigram_table = pow_freqs / pow_freqs.sum()

    def get_target_context_pairs(self, sentences):
        pairs = []
        for sent in sentences:
            ids = [self.word2id[w] for w in sent.split()]
            for i, target_id in enumerate(ids):
                start = max(0, i - self.window_size)
                end = min(len(ids), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((target_id, ids[j]))
        return pairs

    def get_negative_samples(self, count):
        return np.random.choice(self.vocab_size, size=count, p=self.unigram_table)
