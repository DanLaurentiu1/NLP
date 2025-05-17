import os
import pickle
from pathlib import Path

from nltk.corpus import brown
from collections import defaultdict
import numpy as np


# nltk.download('brown')
# nltk.download('universal_tagset')


class HMM:
    def __init__(self, train_data):
        self.train_data = train_data
        self.all_tags = set()
        self.all_words = set()
        self.tag_counts = defaultdict(int)
        self.transition_counts = defaultdict(self._create_default_dict)
        self.emission_counts = defaultdict(self._create_default_dict)
        self.tag2idx = {}
        self.idx2tag = {}
        self.vocab_size = 0
        self.num_tags = 0

        self.preprocess()
        self.create_mappings()
        self.train()

    @staticmethod
    def _create_default_dict():
        return defaultdict(int)

    def preprocess(self):
        for sentence in self.train_data:
            for word, tag in sentence:
                self.all_tags.add(tag)
                self.all_words.add(word.lower())

        self.all_tags = list(self.all_tags)
        self.all_words = list(self.all_words)
        self.all_words.append('<UNK>')
        self.vocab_size = len(self.all_words)
        self.num_tags = len(self.all_tags)

    def create_mappings(self):
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.all_tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

    def train(self):
        for sentence in self.train_data:
            prev_tag = None
            for word, tag in sentence:
                word = word.lower() if word.lower() in self.all_words else '<UNK>'
                self.tag_counts[tag] += 1
                self.emission_counts[tag][word] += 1

                if prev_tag is not None:
                    self.transition_counts[prev_tag][tag] += 1
                prev_tag = tag

    def transition_probability(self, prev_tag, curr_tag):
        return (self.transition_counts[prev_tag][curr_tag] + 1) / (self.tag_counts[prev_tag] + self.num_tags)

    def emission_probability(self, tag, word):
        word = word.lower() if word.lower() in self.all_words else '<UNK>'
        return (self.emission_counts[tag][word] + 1) / (self.tag_counts[tag] + self.vocab_size)

    def viterbi_decode(self, sentence):
        sentence = [word.lower() for word in sentence]
        T = len(sentence)
        N = self.num_tags

        viterbi = np.zeros((N, T))
        backpointers = np.zeros((N, T), dtype=int)

        first_word = sentence[0]
        for i in range(N):
            tag = self.idx2tag[i]
            start_prob = (self.tag_counts[tag] + 1) / (len(self.train_data) + self.num_tags)
            viterbi[i, 0] = start_prob * self.emission_probability(tag, first_word)

        for t in range(1, T):
            for i in range(N):
                max_prob = -np.inf
                best_prev_idx = 0
                curr_tag = self.idx2tag[i]

                for j in range(N):
                    prob = viterbi[j, t - 1] * \
                           self.transition_probability(self.idx2tag[j], curr_tag) * \
                           self.emission_probability(curr_tag, sentence[t])

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_idx = j

                viterbi[i, t] = max_prob
                backpointers[i, t] = best_prev_idx

        best_path = [np.argmax(viterbi[:, -1])]
        for t in range(T - 1, 0, -1):
            best_path.insert(0, backpointers[best_path[0], t])

        return [self.idx2tag[idx] for idx in best_path]


def save_model(model, filename="hmm_pos_tagger.pkl"):
    model_data = {
        'all_tags': list(model.all_tags),
        'all_words': list(model.all_words),
        'tag_counts': dict(model.tag_counts),
        'transition_counts': {k: dict(v) for k, v in model.transition_counts.items()},
        'emission_counts': {k: dict(v) for k, v in model.emission_counts.items()},
        'tag2idx': model.tag2idx,
        'idx2tag': model.idx2tag,
        'vocab_size': model.vocab_size,
        'num_tags': model.num_tags
    }

    with open(filename, "wb") as f:
        pickle.dump(model_data, f)


def load_model(filename="hmm_pos_tagger.pkl"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")

    with open(filename, "rb") as f:
        model_data = pickle.load(f)

    model = HMM.__new__(HMM)

    model.all_tags = set(model_data['all_tags'])
    model.all_words = set(model_data['all_words'])
    model.tag_counts = defaultdict(int, model_data['tag_counts'])
    model.transition_counts = defaultdict(model._create_default_dict,
                                          {k: defaultdict(int, v) for k, v in model_data['transition_counts'].items()})
    model.emission_counts = defaultdict(model._create_default_dict,
                                        {k: defaultdict(int, v) for k, v in model_data['emission_counts'].items()})
    model.tag2idx = model_data['tag2idx']
    model.idx2tag = model_data['idx2tag']
    model.vocab_size = model_data['vocab_size']
    model.num_tags = model_data['num_tags']
    return model


def train_hmm(train_test_split_ratio: float):
    tagged_sentences = brown.tagged_sents(tagset='universal')
    split_idx = int(train_test_split_ratio * len(tagged_sentences))
    train_data = tagged_sentences[:split_idx]
    test_data = tagged_sentences[split_idx:]
    return HMM(train_data), test_data


def prepare_model(train_test_split_ratio: float, train: bool, model_path: str):
    if not train and os.path.exists(model_path):
        return load_model(model_path), brown.tagged_sents(tagset='universal')[int(train_test_split_ratio * len(brown.tagged_sents())):]
    else:
        model, test_data = train_hmm(train_test_split_ratio)
        save_model(model, model_path)
        return model, test_data
