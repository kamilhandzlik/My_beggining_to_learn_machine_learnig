########################################################
# Reproductibility trying to take random out of random #
########################################################


# https://docs.pytorch.org/docs/stable/notes/randomness.html
# https://en.wikipedia.org/wiki/Random_seed
"""
In short how a neural network learns:
start with random numbers -> tensor operations -> update random numbers to try and make them better representation
of the data -> again -> again -> ...

To reduce the randomnes in neural networks and pytorch comes the concept of a random seed
essentialy what random seed does id flavour the randomness
"""
import torch


# Create 2 random tensors
random_tensor_a = torch.rand(3, 4)
random_tensor_b = torch.rand(3, 4)

print(random_tensor_a)
print(random_tensor_b)
print(random_tensor_a == random_tensor_b)
"""
expected output:
tensor([[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]])

or something similar
"""

# Let's make some random by reproductible tensors
# Set random seed

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)  # -> This only works for one block of kode
random_tensor_c = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_d = torch.rand(3, 4)

print(random_tensor_c)
print(random_tensor_d)
print(random_tensor_c == random_tensor_d)
"""
expected output:
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])

or something similar
"""
