import torch

# Tworzenie tensorów i tensoropodobnych (mogłem to źle przetłumaczyć originał to tensor like)
# Pamiętaj arange będzie usunięty w przyszłości

tensor_range = torch.range(0, 10)
print(f"tensor_range: {tensor_range}")

# zakres zaczynasz od zera a kończysz na liczbie, którą chcesz + 1
one_to_ten = torch.range(1, 11)
print(f"one_to_ten: {one_to_ten}")

one_to_ten_arange = torch.arange(start=1, end=10, step=1)
print(f"one_to_ten_arange: {one_to_ten_arange}")

# Tworzenie tensoropodobnych podobny kształt inna zawartość
ten_zeroes = one_to_ten_arange.new_zeros(10)
print(f"ten_zeroes: {ten_zeroes}")
