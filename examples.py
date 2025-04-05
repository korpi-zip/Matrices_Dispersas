# ejemplos.py

from sparse_matrix import TPMMatrix

# Paso 1: Crear una matriz TPM con 3 variables binarias
n = 3
tpm = TPMMatrix(n=n, init_random=False)  # no inicializa con random

# Paso 2: Definir un estado binario específico
estado = [1, 0, 1]  # este es un estado del tiempo t

# Paso 3: Asignar una probabilidad para la variable 2 en t+1
nueva_probabilidad = 0.75
indice_variable = 2  # la tercera variable (C)
tpm.set_probability(estado, indice_variable, nueva_probabilidad)

# Paso 4: Obtener las probabilidades del estado en t+1
probabilidades = tpm.get_next_probabilities(estado)

# Paso 5: Mostrar el resultado
print(f"Probabilidades para el estado {estado} en t+1:")
for i, p in enumerate(probabilidades):
    print(f" - Variable {i}: {p:.2f}")
