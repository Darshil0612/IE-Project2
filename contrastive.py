import numpy as np
import os
import pickle
from collections import defaultdict
import random

NOISE = "<noise>"
data_dir = "C:/Users/DARSHIL/Downloads/Project2/data"


def read_file_line_by_line(file_name, func=lambda x: x, skip_header=True):
    res = []
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()
        for line in fin:
            if line.strip():
                fields = func(line.strip())
                res.append(fields)
    return res


class HMM:
    def __init__(self, num_states, num_outputs):
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.transitions = np.random.dirichlet(np.ones(num_states), size=num_states)
        self.emissions = np.random.dirichlet(np.ones(num_outputs), size=num_states)
        self.initial_probabilities = np.random.dirichlet(np.ones(num_states))

    def initialize_randomly(self):
        self.transitions = np.random.dirichlet(np.ones(self.num_states), size=self.num_states)
        self.emissions = np.random.dirichlet(np.ones(self.num_outputs), size=self.num_states)

    def decode(self, observations):
        T = len(observations)
        N = self.num_states
        V = np.zeros((N, T))
        B = np.zeros((N, T), dtype=int)
        V[:, 0] = self.initial_probabilities * self.emissions[:, observations[0]]
        for t in range(1, T):
            for j in range(N):
                seq_probs = V[:, t - 1] * self.transitions[:, j] * self.emissions[j, observations[t]]
                V[j, t] = np.max(seq_probs)
                B[j, t] = np.argmax(seq_probs)
        last_state = np.argmax(V[:, T - 1])
        path = np.zeros(T, dtype=int)
        path[T - 1] = last_state
        for t in range(T - 2, -1, -1):
            path[t] = B[path[t + 1], t + 1]
        return path

    def update_parameters(self, observations):
        T = len(observations)
        alpha = self.forward(observations)
        beta = self.backward(observations)
        xi = np.zeros((T - 1, self.num_states, self.num_states))
        for t in range(T - 1):
            denominator = np.sum(alpha[t] * beta[t])
            for i in range(self.num_states):
                numerator = alpha[t, i] * self.transitions[i] * self.emissions[:, observations[t + 1]] * beta[t + 1]
                xi[t, i] = numerator / denominator
        gamma = np.sum(xi, axis=2)
        self.transitions = np.sum(xi, axis=0) / np.sum(gamma, axis=0).reshape(-1, 1)
        gamma = np.vstack((gamma, alpha[-1]))
        for k in range(self.num_outputs):
            mask = (observations == k)
            self.emissions[:, k] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)
        self.initial_probabilities = gamma[0] / np.sum(gamma[0])

    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.num_states))
        alpha[0] = self.initial_probabilities * self.emissions[:, observations[0]]

        for t in range(1, T):
            for j in range(self.num_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transitions[:, j]) * self.emissions[j, observations[t]]

        return alpha

    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.num_states))
        beta[-1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.sum(beta[t + 1] * self.transitions[i] * self.emissions[:, observations[t + 1]])

        return beta




class Word_Recognizer:
    def __init__(self, use_full_dataset=True, split_ratio=0.8):
        self.lblnames = read_file_line_by_line(os.path.join(data_dir, "clsp.lblnames"))
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"),
                                              func=lambda x: [int(label) for label in x.split()])
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))
        if not use_full_dataset:
            self.split_data(split_ratio)

    def split_data(self, split_ratio):
        total_samples = len(self.trnlbls)
        indices = list(range(total_samples))
        random.shuffle(indices)
        split_point = int(split_ratio * total_samples)
        self.train_indices = indices[:split_point]
        self.validation_indices = indices[split_point:]

    def get_word_model(self, script):
        hmm = HMM(5, len(set(np.concatenate(self.trnlbls))))
        return hmm

    def train(self, num_epochs):
        best_accuracy = 0
        for epoch in range(num_epochs):
            for i in self.train_indices:
                word = self.trnscr[i]
                labels = self.trnlbls[i]
                word_hmm = self.get_word_model(word)
                word_hmm.update_parameters(labels)
            accuracy = self.validate()
            print(f"Epoch {epoch + 1}: Validation Accuracy = {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save(epoch)
        print(f"Best Accuracy: {best_accuracy}")
        return best_accuracy

    def validate(self):
        correct = 0
        total = 0
        for i in self.validation_indices:
            labels = self.trnlbls[i]
            word = self.trnscr[i]
            word_hmm = self.get_word_model(word)
            predictions = word_hmm.decode(labels)
            correct += sum(p == l for p, l in zip(predictions, labels))
            total += len(labels)
        return correct / total if total > 0 else 0

    def save(self, epoch):
        model_path = os.path.join(data_dir, f"model_epoch_{epoch}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {model_path}")

def main():
    recognizer = Word_Recognizer(use_full_dataset=False, split_ratio=0.8)
    recognizer.train(10)


if __name__ == "__main__":
    main()
