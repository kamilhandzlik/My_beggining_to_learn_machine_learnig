import torch

### Getting information from tensor (tensor attributes)
"""
1. Tensors not right datatype - to do get datatype from a tensor, can use tensor.dtype
2. Tensors not right shape - to get shape from tensors, can use tensors.shape
3. Tensors not on the right device - to get device from a tensor, can use tensor.device
"""

# Create random Tensor in float
some_tensor_1 = torch.rand([3, 4], dtype=torch.float16)
# Create random Tensor in int
some_tensor_2 = torch.randint(low=0, high=10, size=(5, 4), dtype=torch.int32)

# Find out details about some tensor
print(f"Data type of tensor: {some_tensor_1.dtype}")
print(f"Shape of tensor: {some_tensor_1.shape}")
print(f"Device tensor is on: {some_tensor_1.device}")  # cpu or gpu

print(f"Data type of tensor: {some_tensor_2.dtype}")
print(f"Shape of tensor: {some_tensor_2.shape}")
print(f"Device tensor is on: {some_tensor_2.device}")  # cpu or gpu


# Manipulating Tensors (tensor operations)
