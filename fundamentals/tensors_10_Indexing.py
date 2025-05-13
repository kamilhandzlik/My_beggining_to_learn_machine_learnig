##########################################
# Indexing (selecting data from tensors) #
##########################################

# indexing in pytorch is similar to numpy
import torch

x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"\033[32mx: {x}\nx shape: {x.shape}\033[0m")


# Indexing tensor
print(f"\033[33mx[0]: {x[0]}\033[0m")

# Indexing tensor on the middle bracket (dim=1)
print(f"\033[33mx[0][0]: {x[0][0]}\033[0m")

# Indexing tensor on the most inner bracket (last dimension)
print(f"\033[33mx[0][1][1]: {x[0][1][1]}\033[0m")  # -> zwróci 5

"""
Wyjaśnienie indeksowania tutaj mam macierz 3 na 3:
pierwszy [0] wskazuje na pierwsze ([]) a że jest jedno to może wskazywać tylko 1 wymiar jeżeli zmienimy na x[1] będzie error
([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
to x[0][0] wskazuje na ten element [1, 2, 3] i analogicznie to x[0][1] wskazuje na ten element [4, 5, 6].
x[0][2] jest analogiczne gdzie x[0][3] zwróci error bo nie ma już więcej takic elementów.
x[0][0][0] zwraca poszczegulne elementy z [1, 2, 3] tutaj 1 i analogicznie x[0][1][1] zwróci 5
"""

# You can also use ":" to select all of target dimension
x[:, 0]
print(x[:, 0])

# Get all values of 0 and 1 deminsion but only index 1 and 2 dimension
print(x[:, :, 1])

# Get all values of the 0 dimension but only the 1 index value of 1st an 2nd dimension
print(x[:, 1, 1])

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0, 0, :])

# index on x to return 9
print(x[0][2][2])

# index on x to return 3, 6, 9
print(x[:, :, 2])
