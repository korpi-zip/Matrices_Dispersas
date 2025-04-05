# controller.py

from sparse_matrix import TPMMatrix
import numpy as np
import random

def compare_memory_usage(sparse_matrix, dense_matrix):
    sparse_total = sparse_matrix.data.nbytes
    if hasattr(sparse_matrix, 'indices'):
        sparse_total += sparse_matrix.indices.nbytes
    if hasattr(sparse_matrix, 'indptr'):
        sparse_total += sparse_matrix.indptr.nbytes

    dense_total = dense_matrix.nbytes

    print("📊 Comparación de uso de memoria:")
    print(f"   Matriz dispersa: {sparse_total / 1024:.2f} KB")
    print(f"   Matriz densa:    {dense_total / 1024:.2f} KB")
    print("-" * 40)

# Paso 1: Crear TPM con n = 4
n = 4
tpm = TPMMatrix(n=n, init_random=True, seed=42)

print("▶ Estados posibles:")
print(tpm.states)

print("\n▶ Matriz de probabilidades inicial (forma densa):")
tpm.print_dense()

# Comparar uso de memoria antes de modificar
dense = tpm.matrix.toarray()
print("\n🔍 Uso de memoria INICIAL:")
compare_memory_usage(tpm.matrix_csr, dense)

# Paso 2: Reemplazar la mayoría de valores por 0 (simular matriz más dispersa)

# Porcentaje de valores que NO serán cero (por ejemplo, 10%)
sparsity_ratio = 0.10
print(f"\n🧹 Asignando 0 al {(1 - sparsity_ratio) * 100:.0f}% de los valores...")

for i in range(tpm.num_states):
    for j in range(tpm.n):
        if random.random() > sparsity_ratio:
            tpm.set_probability(tpm.states[i], j, 0.0)

# Recalcular CSR y Dense
dense_after = tpm.matrix.toarray()
print("\n▶ Matriz de probabilidades modificada (forma densa):")
tpm.print_dense()

# Comparar uso de memoria después de hacerla dispersa
print("\n🔍 Uso de memoria DESPUÉS de hacer dispersa:")
compare_memory_usage(tpm.matrix_csr, dense_after)
