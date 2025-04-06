# sparse_matrix.py
from scipy import sparse, optimize
import numpy as np
from itertools import product

class TPMMatrix:
    def __init__(self, n, init_random=True, seed=42):
        self.n = n  # número de variables
        self.num_states = 2 ** n
        self.states = list(product([0, 1], repeat=n))
        self.state_to_index = {tuple(state): idx for idx, state in enumerate(self.states)}

        # Crear matriz dispersa en formato LIL para edición eficiente
        self.matrix = sparse.lil_matrix((self.num_states, n), dtype=np.float32)

        if init_random:
            np.random.seed(seed)
            self._init_random()

        # Convertir a CSR para operaciones eficientes
        self.matrix_csr = self.matrix.tocsr()

    def _init_random(self):
        for i in range(self.num_states):
            probs = np.random.rand(self.n)
            probs /= probs.sum()  # normalizar a suma 1
            for j in range(self.n):
                self.matrix[i, j] = probs[j]

    def set_probability(self, state_bin, var_index, value):
        idx = self.state_to_index[tuple(state_bin)]
        self.matrix[idx, var_index] = value
        self.matrix_csr = self.matrix.tocsr()  # actualizar CSR

    def get_next_probabilities(self, state_bin):
        idx = self.state_to_index[tuple(state_bin)]
        return self.matrix_csr.getrow(idx).toarray().flatten()

    def apply_transition_sparse(self, current_state_vector):
        """
        Aplica la transición directamente usando la matriz dispersa en CSR.
        current_state_vector debe ser un numpy array de tamaño (2^n,)
        """
        return self.matrix_csr.transpose() @ current_state_vector

    def normalize_row(self, state_bin):
        """
        Ajusta los valores de una fila para que sumen 1 usando mínimos cuadrados.
        Solo afecta una fila, no la matriz completa.
        """
        idx = self.state_to_index[tuple(state_bin)]
        row = self.matrix_csr.getrow(idx).toarray().flatten()

        def objective(x):
            return np.sum((x - row) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(self.n)]

        result = optimize.minimize(objective, row, bounds=bounds, constraints=constraints)

        if result.success:
            for j in range(self.n):
                self.set_probability(state_bin, j, result.x[j])
        else:
            raise ValueError("No se pudo normalizar la fila.")

    def print_dense(self):
            """
            Imprime la matriz completa en formato denso (útil para depuración).
            """
            print(self.matrix_csr.toarray())