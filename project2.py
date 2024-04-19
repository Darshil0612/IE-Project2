import numpy as np
import matplotlib.pyplot as plt
import string

import os
from collections import Counter
from itertools import chain
from scipy.special import logsumexp
from collections import defaultdict
from tqdm import tqdm
import pickle

NOISE = "<noise>"
data_dir = "C:/Users/DARSHIL/Downloads/Project2/data"

def read_file_line_by_line(file_name, func=lambda x: x, skip_header=True):
    print("reading file: %s" % file_name)
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()
        for line in fin:
            if len(line.strip()) == 0:
                continue
            fields = func(line.strip())
            res.append(fields)
    print("%d lines, done" % len(res))
    return res

class HMM:

    def __init__(self, num_states, num_outputs):
        self.states = range(num_states)
        self.outputs = range(num_outputs)
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.transitions = None
        self.emissions = None
        self.non_null_arcs = []
        self.null_arcs = dict()
        self.topo_order = []
        self.output_arc_counts = None
        self.output_arc_counts_null = None
        self.null_transitions = np.zeros(num_states)

    def init_transition_probs(self, transitions, null_transitions=None):
        self.transitions = transitions
        if null_transitions is not None:
            self.null_transitions = null_transitions
        self._assert_transition_probs()

    def init_emission_probs(self, emission):
        if emission.shape != (self.num_states, self.num_outputs):
            raise ValueError(
                f"Emission probabilities must be shape ({self.num_states}, {self.num_outputs}), got {emission.shape}")
        self.emissions = np.array(emission)
        self.emissions /= self.emissions.sum(axis=1, keepdims=True)  # Normalize
        self._assert_emission_probs()

    def init_null_arcs(self, null_arcs=None):
        if null_arcs is not None:
            self.null_arcs = null_arcs
        count = np.zeros(self.num_states)
        for ix in self.null_arcs.keys():
            for iy in self.null_arcs[ix].keys():
                if iy < self.num_states:
                    count[iy] += 1
        stack = [s for s in self.states if count[s] == 0]
        while len(stack) > 0:
            s = stack.pop()
            self.topo_order.append(s)
            if s not in self.null_arcs:
                continue
            for s_y in self.null_arcs[s]:
                count[s_y] -= 1
                if count[s_y] == 0:
                    stack.append(s_y)
        assert len(self.topo_order) == self.num_states

    def add_null_arc(self, ix, iy, prob):
        if self.null_arcs is None:
            self.null_arcs = dict()

        if ix not in self.null_arcs:
            self.null_arcs[ix] = dict()

        self.null_arcs[ix][iy] = prob

    def map_observation(self, obs_value):
        mapped_value = obs_value % self.num_outputs
        print(f"Mapping observation {obs_value} to {mapped_value} (num_outputs: {self.num_outputs})")
        return mapped_value

    def _assert_emission_probs(self):
        emission_sum = self.emissions.sum(axis=1)
        print("Emission sums: ", emission_sum)
        assert np.allclose(emission_sum, 1, atol=1e-6)

    def _assert_transition_probs(self):
        for s in range(self.num_states):
            total_sum = self.transitions[s, :].sum() + self.null_transitions[s]
            assert np.isclose(total_sum, 1, atol=1e-6), f"Transition probabilities must sum to 1 for state {s}"

    def forward(self, data, init_prob=None):
        alphas = np.zeros((len(data) + 1, self.num_states))
        Q = np.ones(len(data) + 1)

        if init_prob is None:
            init_prob = np.full(self.num_states, 1.0 / self.num_states)
        alphas[0] = init_prob
        Q[0] = np.sum(alphas[0])
        alphas[0] /= Q[0]

        for t in range(1, len(data) + 1):
            obs_idx = self.map_observation(data[t - 1])
            print(f"obs_idx for t={t}: {obs_idx}, valid range: 0-{self.num_outputs - 1}")

            for i in range(self.num_states):
                print(f"Current state index: {i}, Total states: {self.num_states}")
                print(f"Emissions matrix shape: {self.emissions.shape}")

                if i >= self.emissions.shape[0]:  # This check ensures you do not exceed bounds
                    raise IndexError(
                        f"State index {i} is out of bounds for emissions matrix with shape {self.emissions.shape}")

                alphas[t, i] = np.sum(alphas[t - 1] * self.transitions[:, i] * self.emissions[i, obs_idx])

            Q[t] = np.sum(alphas[t])
            alphas[t] /= Q[t]

        return alphas, Q

    def backward(self, data, Q, init_beta=None):
        betas_ = np.zeros((len(data) + 1, self.num_states))
        if init_beta is None:
            betas_[-1] = np.ones(self.num_states)
        else:
            betas_[-1] = init_beta
        betas_[-1] /= Q[-1]

        for t in range(len(data) - 1, -1, -1):
            obs_index = self.map_observation(data[t])
            for j in range(self.num_states):
                betas_[t][j] = np.dot(betas_[t + 1] * self.emissions[:, obs_index], self.transitions[:, j])
            betas_[t] /= Q[t]

        return betas_

    def backward_log(self, data, init_log_beta=None):
        log_betas = np.empty((len(data) + 1, self.num_states))

        if init_log_beta is None:
            log_betas[-1] = np.zeros(self.num_states)
        else:
            log_betas[-1] = init_log_beta

        for t in range(len(data) - 1, -1, -1):
            obs = data[t]
            for j in range(self.num_states):
                log_betas[t][j] = logsumexp(
                    log_betas[t + 1] + np.log(self.emissions[obs][j]) + np.log(self.transitions[j]))

            for s in reversed(self.topo_order):
                if s not in self.null_arcs:
                    continue
                for s_y in self.null_arcs[s]:
                    log_betas[t][s] = logsumexp([
                        log_betas[t][s],
                        log_betas[t][s_y] + np.log(self.null_arcs[s][s_y])
                    ])
        return log_betas

    def un_norm_alphas_(self, alphas_, Q):
        alphas = np.copy(alphas_)
        cur_q = 1
        for t in range(alphas.shape[0]):
            cur_q *= Q[t]
            alphas[t] *= cur_q
        return alphas

    def un_norm_betas_(self, betas_, Q):
        betas = np.copy(betas_)
        cur_q = 1
        for t in range(betas.shape[0] - 2, -1, -1):
            cur_q *= Q[t + 1]
            betas[t] *= cur_q
        return betas


    def concatenate(self, other_hmm):
        if self.emissions.shape[1] != other_hmm.emissions.shape[1]:
            raise ValueError(
                f"Emission dimension mismatch: {self.emissions.shape[1]} and {other_hmm.emissions.shape[1]}")

        new_num_states = self.num_states + other_hmm.num_states
        new_transitions = np.zeros((new_num_states, new_num_states))
        new_null_transitions = np.zeros(new_num_states)

        new_transitions[:self.num_states, :self.num_states] = self.transitions
        new_null_transitions[:self.num_states] = self.null_transitions

        new_transitions[self.num_states:, self.num_states:] = other_hmm.transitions
        new_null_transitions[self.num_states:] = other_hmm.null_transitions

        new_emissions = np.vstack([self.emissions, other_hmm.emissions])

        self.transitions = new_transitions
        self.null_transitions = new_null_transitions
        self.emissions = new_emissions
        self.num_states = new_num_states

        print(f'Concatenated HMM: new shape {self.emissions.shape}')


    def forward_backward(self, train, init_prob=None, init_beta=None, update_params=True):
        alphas_, Q = self.forward(train, init_prob=init_prob)
        betas_ = self.backward(train, Q, init_beta=init_beta)
        self.reset_counters()

        for t in range(1, len(train) + 1):
            obs = train[t - 1]
            step1 = np.zeros((self.num_states, self.num_states))
            for i in range(self.num_states):
                for j in range(self.num_states):
                    step1[i, j] = alphas_[t - 1, i] * self.transitions[i, j] * betas_[t, j] * self.emissions[j, obs]

            step1_sum = np.sum(step1)
            if not np.isclose(step1_sum, 1.0, atol=1e-6):
                step1 /= step1_sum
            assert np.isclose(np.sum(step1), 1.0, atol=1e-6), "Probabilities do not sum to 1 after normalization"

            self.output_arc_counts[obs] += step1

            for ix in self.null_arcs.keys():
                for iy in self.null_arcs[ix].keys():
                    p = alphas_[t][ix] * self.null_arcs[ix][iy] * betas_[t][iy] * Q[t]
                    self.output_arc_counts_null[ix][iy] += p

        if update_params:
            self.update_params()

        log_likelihood = self.log_likelihood(alphas_, betas_, Q)
        return log_likelihood

    def reset_counters(self):
        self.output_arc_counts = np.zeros((self.num_outputs, self.num_states, self.num_states))
        self.output_arc_counts_null = defaultdict(lambda: defaultdict(lambda: 0))

    def set_counters(self, another_output_arc_counts, another_output_arc_counts_null):
        self.output_arc_counts += another_output_arc_counts

        for ix in another_output_arc_counts_null.keys():
            for iy in another_output_arc_counts_null[ix].keys():
                self.output_arc_counts_null[ix][iy] += another_output_arc_counts_null[ix][iy]

    def update_params(self):
        if np.any(self.output_arc_counts.sum(axis=(0, 1)) > 0):
            total_emission_counts = self.output_arc_counts.sum(axis=0)
            total_emission_counts_sum = total_emission_counts.sum(axis=1, keepdims=True) + 1e-10
            self.emissions = total_emission_counts / total_emission_counts_sum
            self.emissions = np.nan_to_num(self.emissions / self.emissions.sum(axis=1, keepdims=True))

        for s in range(self.num_states):
            null_sum = sum(self.null_arcs.get(s, {}).values())
            transition_sum = self.transitions[s, :].sum()
            total = transition_sum + null_sum

            if total > 0:
                self.transitions[s, :] /= total
                if s in self.null_arcs:
                    for s_y in self.null_arcs[s]:
                        self.null_arcs[s][s_y] /= total

        self.transitions = np.nan_to_num(self.transitions)
        self._assert_emission_probs()
        self._assert_transition_probs()

    def log_likelihood(self, alphas_, betas_, Q):
        return np.log((alphas_[-1] * betas_[-1] * Q[-1]).sum()) + np.log(Q).sum()

    def compute_log_likelihood(self, data, init_prob, init_beta):
        alphas_, Q = self.forward(data, init_prob=init_prob)
        return np.log((alphas_[len(data)] * init_beta).sum()) + np.log(Q).sum()

class Word_Recognizer:

    def __init__(self, restore_ith_epoch=None):
        self.lblnames = read_file_line_by_line(os.path.join(data_dir, "clsp.lblnames"))
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"), func=lambda x: x.split())
        self.endpts = read_file_line_by_line(os.path.join(data_dir, "clsp.endpts"), func=lambda x: list(map(int, x.split())))
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))
        self.devlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.devlbls"), func=lambda x: x.split())
        self.train_words = set(self.trnscr)

        assert len(self.trnlbls) == len(self.endpts)
        assert len(self.trnlbls) == len(self.trnscr)

        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.noise_id = len(self.letters)
        self.letters.append(NOISE)
        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})

        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})

        self.trnlbls = [[self.label2id[lbl] for lbl in line] for line in self.trnlbls]
        self.devlbls = [[self.label2id[lbl] for lbl in line] for line in self.devlbls]
        self.trnscr = [[self.letter2id[c] for c in word] for word in self.trnscr]

        lbl_freq = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=0)
        lbl_freq_noise = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=1, endpts=self.endpts)

        self.letter_id2hmm = self.init_letter_hmm(lbl_freq, lbl_freq_noise, self.id2letter)
        self.sil_hmm = self.init_sil_hmm()

    def get_unigram(self, trnlbls, nlabels, smooth=0, endpts=None):
        counter = Counter(chain.from_iterable(trnlbls))
        total_count = sum(counter.values()) + smooth * nlabels
        freq = np.array([(counter[i] + smooth) / total_count for i in range(nlabels)])
        return freq

    def init_sil_hmm(self):
        num_states = 5
        num_outputs = 256
        transition_probs = np.array([
            [0.25, 0.25, 0.25, 0.25, 0.00],
            [0.00, 0.25, 0.25, 0.25, 0.25],
            [0.00, 0.25, 0.25, 0.25, 0.25],
            [0.00, 0.25, 0.25, 0.25, 0.25],
            [0.00, 0.00, 0.00, 0.00, 0.75]
        ])
        null_transitions = np.array([0, 0, 0, 0, 0.25])

        sil_hmm = HMM(num_states, num_outputs)
        sil_hmm.init_transition_probs(transition_probs, null_transitions)
        return sil_hmm


    def init_letter_hmm(self, lbl_freq, lbl_freq_noise, id2letter):
        letter_id2hmm = {}
        num_states = 3
        num_outputs = 256

        for letter_id, letter in id2letter.items():
            transition_probs = np.array([
                [0.8, 0.2, 0.0],
                [0.0, 0.8, 0.2],
                [0.0, 0.0, 1.0]
            ])
            emission_probs = np.full((num_states, num_outputs), 1.0 / num_outputs)
            if letter == NOISE:
                emission_probs = np.tile(lbl_freq_noise, (num_states, 1))

            emission_probs /= np.sum(emission_probs, axis=1, keepdims=True)

            hmm = HMM(num_states, num_outputs)
            hmm.init_transition_probs(transition_probs)
            hmm.init_emission_probs(emission_probs)
            letter_id2hmm[letter_id] = hmm

            print(f'Initialized HMM for {letter}: shape {hmm.emissions.shape}')  # Debugging output

        return letter_id2hmm

    def id2word(self, w):
        return ''.join(map((lambda c: self.id2letter[c]), w))

    def get_word_model(self, scr):
        h = self.letter_id2hmm[
            self.letter2id[NOISE]]

        for char_id in scr:
            letter_hmm = self.letter_id2hmm[char_id]
            h.concatenate(letter_hmm)

        h.concatenate(self.letter_id2hmm[self.letter2id[NOISE]])

        return h

    def update_letter_counters(self, scr, word_hmm):

        current_state_index = 0

        for char_id in scr:
            letter_hmm = self.letter_id2hmm[char_id]
            num_states = letter_hmm.num_states


            transition_counts, emission_counts = word_hmm.extract_counts(start=current_state_index, length=num_states)

            letter_hmm.update_counts(transition_counts, emission_counts)

            current_state_index += num_states


    def train(self, num_epochs):
        sorted_indices = sorted(range(len(self.trnscr)), key=lambda i: self.trnscr[i])
        trnlbls_sorted = [self.trnlbls[i] for i in sorted_indices]
        trnscr_sorted = [self.trnscr[i] for i in sorted_indices]

        for epoch in range(num_epochs):
            total_log_likelihood = 0
            for word_id, (word, labels) in enumerate(zip(trnscr_sorted, trnlbls_sorted)):
                word_hmm = self.get_word_model(word)
                log_likelihood = word_hmm.forward_backward(labels)
                total_log_likelihood += log_likelihood
            avg_log_likelihood = total_log_likelihood / len(trnscr_sorted)
            print(f"Epoch {epoch + 1}: Avg Log Likelihood = {avg_log_likelihood}")
            self.save(epoch)
            self.test()

    def test(self):
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})

        word_likelihoods = np.zeros((len(words2id), len(self.devlbls)))

        for sample_index, labels in enumerate(self.devlbls):
            for word, word_id in words2id.items():
                word_model = self.get_word_model([self.letter2id[char] for char in word if char in self.letter2id])
                word_likelihoods[word_id, sample_index] = word_model.forward(labels)

        most_likely_words_indices = word_likelihoods.argmax(axis=0)
        most_likely_words = [id2words[index] for index in most_likely_words_indices]

        return most_likely_words, word_likelihoods

        result = word_likelihoods.argmax(axis=0)
        result = [id2words[res] for res in result]

    def save(self, i_epoch):
        fn = os.path.join(data_dir, "%d.mdl.pkl" % i_epoch)
        print("Saved to:", fn)
        # for letter_id, hmm in self.letter_id2hmm.items():
            # hmm.output_arc_counts = None
            # hmm.output_arc_counts_null = None
        pickle.dump(self.letter_id2hmm, open(fn, "wb"))

    def load(self, i_epoch):
        return pickle.load(open(os.path.join(data_dir, "%d.mdl.pkl" % i_epoch), "rb"))


def main():
    n_epochs = 100
    wr = Word_Recognizer()
    wr.train(num_epochs=n_epochs)

if __name__ == '__main__':
    main()