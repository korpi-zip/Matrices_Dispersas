# sparse_matrix.py

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

class TPMMatrix:
    def __init__(self, n, init_random=True, seed=None):
        self.n = n
        self.num_states = 2 ** n
        self.states = self._generate_binary_states(n)
        self.matrix = lil_matrix((self.num_states, n), dtype=np.float32)

        if init_random:
            if seed is not None:
                np.random.seed(seed)
            for i in range(self.num_states):
                for j in range(n):
                    self.matrix[i, j] = np.random.rand()

        self.matrix_csr = self.matrix.tocsr()

    def _generate_binary_states(self, n):
        return np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2 ** n)])

    def state_to_index(self, state_vector):
        return int("".join(map(str, state_vector)), 2)

    def get_next_probabilities(self, state_vector):
        index = self.state_to_index(state_vector)
        return self.matrix_csr.getrow(index).toarray().flatten()

    def set_probability(self, state_vector, variable_index, probability):
        index = self.state_to_index(state_vector)
        self.matrix[index, variable_index] = probability
        self.matrix_csr = self.matrix.tocsr()  # refrescar CSR

    def print_dense(self):
        print(self.matrix.toarray())
