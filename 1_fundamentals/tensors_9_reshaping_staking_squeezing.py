############################################
# Reshaping Stacking Squeezing unsqueezing #
############################################

import torch

"""
* Reshaping -> reshapes an input tensor to a defined shape
* View      -> return a view of an input tensor of certain shape but keep the same memory as the original tensor
* Stacking  -> combine multiple tensors on top of each other vstack or side by side hstack
* Squezze   -> removes all 1 dimensions from a tensor
* Usqueeze  -> add a 1 dimensionn to a target tensor
* Permute   -> return a view of the input with dimensions permuted swapped in certain way
"""

# Create tensor
x = torch.arange(1.0, 10.0)
print(x, x.shape)
"""
Output:
tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.Size([9])
"""

# Add an extra dimension
# x_reshaped = x.reshape(1, 7) -> this will return error because number of elements does not match
x_reshaped = x.reshape(1, 9)  # -> this will work x has 9 elements 1 * 9 = 9
x_reshaped_2 = x.reshape(9, 1)  # -> this will work x has 9 elements 1 * 9 = 9
x_reshaped_3 = x.reshape(3, 3)  # -> this will work x has 9 elements 3 * 3 = 9
print(f"Reshape: \n {x_reshaped, x_reshaped.shape}\n")
print(f"Reshape 2:\n {x_reshaped_2, x_reshaped_2.shape}\n")
print(f"Reshape 3:\n {x_reshaped_3, x_reshaped_3.shape}\n")

# Change the view
z = x.view(1, 9)
print(f"Changed view: \n {z, z.shape}\n")


# Zmiana z zmienia x ponieważ view tensora dzieli te samo miejsce w pamięci co oryginalny input
z[:, 0] = 5
print(f"z: {z},\n x: {x}\n")


# Stack tensos on top of each other
x_stacked = torch.stack([x, x, x, x], dim=1)
print(f"x stacked {x_stacked}\n")


# Squeeze
x_squeezed = x.squeeze()
print(f"\033[32mx reshaped: {x_reshaped}\033[0m\n")
print(f"\033[32mShape of x reshaped: {x_reshaped.shape}\033[0m\n")
print(f"Squeezing x:\n{x_squeezed}\n Shape: {x_squeezed.shape}\n")


# torch.usqueeze() - adds a single dimension to a target tensor at a specified dim (dimension)
# print(f"\033[32m\033[0m]")
print(f"\033[32mPrevious target: {x_squeezed}\033[0m]")
print(f"\033[32mPrevious shape: {x_squeezed.shape}\033[0m]\n")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\033[33mNew target: {x_unsqueezed}\033[0m]")
print(f"\033[33mNew shape: {x_unsqueezed.shape}\033[0m]\n")

x_unsqueezed_dim_1 = x_squeezed.unsqueeze(dim=1)
print(f"\033[33mNew target: {x_unsqueezed_dim_1}\033[0m]")
print(f"\033[33mNew shape: {x_unsqueezed_dim_1.shape}\033[0m]\n")


# Permute = permutacja zmiana wymiarów macierzy/tensora najczęściej używane przy obrazach, images
x_original = torch.rand(size=(224, 224, 3))  # height, width, colour_channels

# Permute original tensor to rearrange the axis or dim order
x_permuted = x_original.permute(2, 0, 1)  # colour_channels, height, width
print(f"\033[32mPrevious target: {x_original}\033[0m]")
print(f"\033[32mPrevious shape: {x_original.shape}\033[0m]\n")
print(f"\033[33mNew target: {x_permuted}\033[0m]")
print(f"\033[33mNew shape: {x_permuted.shape}\033[0m]\n")
