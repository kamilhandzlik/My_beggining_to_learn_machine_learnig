import torch


# Zera i jedynki
# tensor samych zer
zeros = torch.zeros(size=(3, 4))
print(f"Zeros:\n{zeros}\n")

# tensor samych jedynek
ones = torch.ones(size=(3, 4))
print(f"Ones:\n{ones}\n")

# typ danych
print(f"Zeros dtype: {zeros.dtype}")
print(f"Ones dtype: {ones.dtype}\n")
