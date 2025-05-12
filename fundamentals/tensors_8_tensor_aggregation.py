########################################
# Finding the min, max, mean, sum, etc #
########################################
import torch

# Create tensor
x = torch.arange(0, 100, 10)
print(f"x: {x}\n")

# Find min
print(f"torch.min(x): {torch.min(x)} also works x.min() {x.min()} \n")

# Find max
print(f"torch.mxa(x): {torch.max(x)} also works x.max() {x.max()} \n")

# Find mean
# print(torch.mean(x)) -> this will not work because dataype is wrong
print(torch.mean(x.type(torch.float32)))  # -> but this wiil work great :)
# torch mean wymaga użycia float 32

# Find sum
print(f"torch.sum(x): {torch.sum(x)} also works x.sum() {x.sum()} \n")

# Finding positional min and max
print(f"finding position in tensor with minimal value: {x.argmin()}")
print(f"finding position in tensor with maximal value: {x.argmax()} \n")
