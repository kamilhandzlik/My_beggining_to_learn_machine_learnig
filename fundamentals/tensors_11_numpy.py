#############################
## Pytroch tensors & numpy ##
#############################
import torch
import numpy as np

"""
Since NumPy is a popular Python numerical computing library, PyTorch has functionality to interact with it nicely.

The two main methods you'll want to use for NumPy to PyTorch (and back again) are:

torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.
torch.Tensor.numpy() - PyTorch tensor -> NumPy array.
"""

# Numpay array to tensor

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)

"""
Output:
[1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)
Używanie tej metody podstawowo zwraca float64 można to zmienić samemu ustalając dtype
"""

# Change the value of array
array = array + 1
print(array, tensor)

# Tensor to Numpy array
tensor_1 = torch.ones(7)
numpy_tensor = tensor_1.numpy()
print(tensor_1, numpy_tensor)


# Change the tensor, what happens to numpy_tensor?
tensor_1 += 1
print(tensor_1, numpy_tensor)
