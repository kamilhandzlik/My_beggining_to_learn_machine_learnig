# Manipulating Tensors (tensor operations)
import torch


"""
Tensor operations include:
- dodawanie
- odjmowanie
- mnożenie (elementów)
- dzielenie
- mnożenie (macierzy)
"""

# Create tensor and add 10 to it
tensor = torch.tensor([1, 2, 3])
print(f"tensor + 10 {tensor + 10}")
print(f"tensor.add(10) {tensor.add(10)}")

# Multiplying tensor by 10
print(f"tensor * 10 {tensor * 10}")
print(f"tensor.mul(10) {tensor.mul(10)}")
print(f"tensor {tensor}")
tensor *= 10
print(f"tensor *= 10  {tensor}")

# Subtract
print(f"tensor - 10 {tensor - 10}")
print(f"tensor.sub(10) {tensor.sub(10)}")

# divide
print(f"tensor / 2 {tensor / 2}")
print(f"tensor.div(2) {tensor.div(2)}")
