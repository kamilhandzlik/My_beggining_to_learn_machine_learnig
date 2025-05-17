import torch
import numpy as np

"""
I learn with this course:
https://github.com/mrdbourke/pytorch-deep-learning
https://www.learnpytorch.io/
"""


# scalar/skalar tak samo jak w matematyce inaczej liczby skalarne operujesz normalnie
scalar = torch.tensor(7)
print(f"Scalar: {scalar}")

# scalar dimensions/wymiary przestrzennne skalara
scalar.ndim
print(f"Scalar dimensions: {scalar.ndim}")

# Get tensor back as Python int
scalar.item()
print(f"Scalar as Python int: {scalar.item()}")

# vector/wektor rachunek wektorowy
vector = torch.tensor([7, 7])
print(f"Vector: {vector}")
print(f"Vector dimensions: {vector.ndim}")
print(f"Vector shape: {vector.shape}")

# matrix/macierz operacje na macierzach
MATRIX = torch.tensor([[7, 8], [9, 10]])  # macierz 2 na 2
print(f"Matrix: {MATRIX}")
print(f"Matrix dimensions: {MATRIX.ndim}")
print(f"Matrix shape: {MATRIX.shape}")


# Tensor
TENSOR = torch.tensor([[1, 2, 3], [3, 6, 9], [2, 4, 5]])
print(f"Tensor: {TENSOR}")
print(f"Tensor dimension: {TENSOR.ndim}")
print(f"Tensor shape: {TENSOR.shape}")
print(f"Tensor 0: {TENSOR[0]}")

# ważne pamiętaj o odpowiednich nawiasach zobacz jak różnią ci się wyniki
TENSOR_2 = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
print(f"Tensor 2: {TENSOR_2}")
print(f"Tensor 2 dimension: {TENSOR_2.ndim}")
print(f"Tensor 2 shape: {TENSOR_2.shape}")
print(f"Tensor 2 0: {TENSOR_2[0]}")


# skalary i wektory zapisujemy małymi literami 'scalara' 'vector'
# Macierze i Tensory zapisujemy dużymi literami 'TENSOR' 'MATRIX'

"""
Dlaczego używamy tensorów o przypadkowych elementach/przypadkowej liczbie elementów?

Przypadkowe tensory są ważne ze względu na sposób działania wielu sieci nerunowych i tego jak te sieci się uczą.
Sieci te często zaczynają z przypdakowymi numerami a następnie dostosowują te numery by te lepiej reprezentowały dane.

Zaczyna z przypakowymi numerami -> patrzy na dane -> dostosowuje numery by lepiej reprezentowały dane -> powtórz kroki 2 i 3 
"""

# Tworzenie przypadkowego Tensora 3 na 4
random_tensor = torch.rand(3, 4)
print(f"Random Tensor: {random_tensor}")
print(f"Random Tensor dimensions: {random_tensor.ndim}")
print(f"Random Tensor shape: {random_tensor.shape}")

# Tworzenie przypadkowego Tensora 10 na 10 na 10
# random_tensor_2 = torch.rand(10, 10, 10)
# print(f"Random Tensor 2: {random_tensor_2}")
# print(f"Random Tensor 2 dimensions: {random_tensor_2.ndim}")
# print(f"Random Tensor 2 shape: {random_tensor_2.shape}")


# Tworzenie przypadkowego tensora z kształtem podobnm do image tensor
random_image_size_tensor = torch.rand(224, 224, 3)  # wysokość, szerokość, kolor R G B
print(f"Random Image Size Tensor: {random_image_size_tensor}")
print(f"Random Image Size Tensor dimensions: {random_image_size_tensor.ndim}")
print(f"Random Image Size Tensor shape: {random_image_size_tensor.shape}")
