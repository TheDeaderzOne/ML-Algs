import numpy as np
from typing import List


def max_mult(start_vec: np.ndarray, a: np.ndarray, b: np.ndarray):
    if len(a[0]) != len(b):
        raise ValueError("There is a shape mismatch")

    new_arr = np.zeros((len(a), len(b[0])))
    s_arr = np.zeros((len(a), len(b[0])), dtype=int)
    s_arr[:, 0] = -1  # back_pointers
    new_arr[:, 0] = start_vec

    for y in range(1, len(b[0])):
        for x in range(len(a)):
            new_arr[x][y] = np.max(new_arr[:, y - 1] * a[:, x] * b[x][y])
            s_arr[x][y] = int(np.argmax(new_arr[:, y - 1] * a[:, x] * b[x][y]))

    return new_arr, s_arr


class HMM:
    # encoding must be from 0 to n-1
    def __init__(self, markov_matrix, emission_matrix, encoding: List[str], hidden_encoding: List[str]):

        self.markov_matrix = np.asarray(markov_matrix)
        self.emission_matrix = np.asarray(emission_matrix)

        if len(list(set(encoding))) != self.emission_matrix.shape[1]:
            raise ValueError("Must be one unique encoding for each observed state")

        if len(list(set(hidden_encoding))) != self.markov_matrix.shape[0]:
            raise ValueError("Must be one unique encoding for each hidden state")

        # please just give a list with no duplicates
        self.encoding = {item: index for index, item in enumerate(encoding)}
        self.hidden_encoding = hidden_encoding
        self.hidden_encoding_dict = {item: index for index, item in enumerate(hidden_encoding)}

    def __eig_taker(self, transposed_matr):
        eigenvalues, eigenvectors = np.linalg.eig(transposed_matr)
        eigenvector = np.abs(eigenvectors[:, np.argmax(np.abs(eigenvalues))])
        eigenvector = eigenvector / np.sum(eigenvector)
        return eigenvector

    def likelihood(self, hidden_states, observations, init_prob=None) -> float:

        observ_matr = np.column_stack([self.emission_matrix[:, self.encoding[o]].reshape(-1, 1) for o in observations])
        new_hidden_states = [self.hidden_encoding_dict[o] for o in hidden_states]

        if init_prob is None:
            init_prob_eigenvec = self.__eig_taker(self.markov_matrix.T) * observ_matr[:, 0]
        else:
            init_prob_eigenvec = init_prob * observ_matr[:, 0]

        init_like = init_prob_eigenvec[new_hidden_states[0]]

        for x in range(len(new_hidden_states) - 1):
            init_like *= (
                self.markov_matrix[new_hidden_states[x]][new_hidden_states[x + 1]]
                * observ_matr[new_hidden_states[x + 1]][x + 1]
            )

        return init_like

    # Must be given in Ordinal Encoding or something, for emission matrix
    def decode(self, observations, initial_probabilities=None):

        observ_matr = np.column_stack([self.emission_matrix[:, self.encoding[o]].reshape(-1, 1) for o in observations])

        if initial_probabilities is None:
            init_prob_eigenvec = self.__eig_taker(self.markov_matrix.T) * observ_matr[:, 0]
        else:
            init_prob_eigenvec = initial_probabilities * observ_matr[:, 0]

        mult_object = max_mult(init_prob_eigenvec, self.markov_matrix, observ_matr)

        score = np.max(mult_object[0][:, -1])

        best_row_pointer = np.argmax(mult_object[0][:, -1])

        best_row = [0] * len(observations)

        for j in range(len(observations) - 1, -1, -1):
            best_row[j] = int(best_row_pointer)
            if best_row_pointer >= 0:
                best_row_pointer = mult_object[1][best_row_pointer][j]

        hidden_states = [self.hidden_encoding[x] for x in best_row]

        return hidden_states, score
