# ejemplo.py
import numpy as np
from sparse_matrix import TPMMatrix

# Crear una matriz TPM de 3 variables
n = 3
tpm = TPMMatrix(n)

# Mostrar las probabilidades originales de un estado
estado = [1, 0, 1]
print("Probabilidades originales del estado [1, 0, 1]:")
print(tpm.get_next_probabilities(estado))

# Modificar explícitamente la fila correspondiente al estado [1, 0, 1]
tpm.set_probability(estado, 0, 0.5)  # Asignar 0.5 a la primera variable
tpm.set_probability(estado, 1, 0.3)  # Asignar 0.3 a la segunda variable
tpm.set_probability(estado, 2, 0.8)  # Asignar 0.8 a la tercera variable

print("\nProbabilidades modificadas del estado [1, 0, 1]:")
print(tpm.get_next_probabilities(estado))

# Normalizar explícitamente la fila correspondiente a [1, 0, 1]
tpm.normalize_row(estado)

print("\nProbabilidades normalizadas del estado [1, 0, 1]:")
print(tpm.get_next_probabilities(estado))

# Crear un vector de estado actual (uniforme)
current_state_vector = np.ones(2**n) / (2**n)

# Aplicar la transición dispersa
next_state = tpm.apply_transition_sparse(current_state_vector)

print("\nResultado de la transición usando apply_transition_sparse:")
print(next_state)
