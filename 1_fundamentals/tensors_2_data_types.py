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
"""
Zatem typ danych używany domyślnie przez pyTorch to float32.
Jeśli chcemy używać innego typu danych, możemy to zrobić w następujący sposób:
"""
# Tensor z typem danych float64
tensor_float64 = torch.zeros(size=(3, 4), dtype=torch.float64)
print(f"Tensor float64:\n{tensor_float64}\n")

# Tensor z typem danych int32
tensor_int32 = torch.ones(size=(3, 4), dtype=torch.int32)
print(f"Tensor int32:\n{tensor_int32}\n")

# Zmiana typu danych istniejącego tensora
tensor = torch.tensor([1.0, 2.0, 3.0])
tensor_int = tensor.to(dtype=torch.int32)
print(f"Tensor int:\n{tensor_int}\n")
