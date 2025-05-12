#############################
### Matrix multiplication ###
#############################
# Two main ways of performing multiplication in neural networks and deep learning
# https://www.mathsisfun.com/algebra/matrix-multiplying.html
"""One of the most common errors in deep learnig is shape errors
 1. The **inner dimensions** must much
 * `(3. 2) @ (3, 2)` won't work
 * `(2, 3) @ (3, 2)` will work
 * `(3, 2) @ (2, 3)` will work

 2. The resulting matrix has the shape of the  **outer dimensions**:
* `(2, 3) @ (3, 2)` -> (2, 2)
* `(3, 3) @ (3, 2)` -> (3, 3)
"""
import torch


###################################
# 1. Element-wise multiplication. #
###################################
tensor = torch.tensor([2, 3, 4])
print(f"tensor, '*', tensor {tensor, '*', tensor}")
print(f"Equals: {tensor * tensor}")


#############################
# 2. Matrix multiplication. #
#############################
torch.matmul(tensor, tensor)
# or
tensor @ tensor
print(f"torch.matmul(tensor, tensor): {torch.matmul(tensor, tensor)}")
print(f"tensor @ tensor {tensor @ tensor}")

# Matrix multiplication by hand.
2 * 2 + 3 * 3 + 4 * 4
print(f"Matrix multiplication by hand. in this case 2*2 + 3*3 + 4*4: {2*2 + 3*3 + 4*4}")
# matmul is faster by the way


# time miliseconds 1/1 000
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value)

# time microseconds 1/1 000 000
torch.matmul(tensor, tensor)


# Example for the **inner dimensions** must much
# Won't work
# print(
# f"torch.matmul(torch.rand(3, 2), torch.rand(3, 2)) {torch.matmul(torch.rand(3, 2), torch.rand(3, 2))}"
# )
# will work
print(
    f"torch.matmul(torch.rand(3, 2), torch.rand(2, 3)) {torch.matmul(torch.rand(3, 2), torch.rand(2, 3))}"
)
# will work
print(
    f"torch.matmul(torch.rand(2, 3), torch.rand(2, 3)) {torch.matmul(torch.rand(3, 2), torch.rand(2, 3))}"
)
print(
    f"torch.matmul(torch.rand(3, 2), torch.rand(2, 3)).shape {torch.matmul(torch.rand(3, 2), torch.rand(2, 3)).shape}"
)

# One of the most common errors in deep learnig is shape errors
tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_B = torch.tensor([[7, 11], [21, 15], [9, 6]])


# torch.mm(tensor_A, tensor_B) jest dokładnie tym samym co matmul
# print(torch.matmul(tensor_A, tensor_B)) -> zwróci error (3, 2) @ (3, 2)
# żeby naprawić kształt możemy użyć transpose żeby zmienić osie lub wymiary tensora
print(f"tensor_B original {tensor_B}")
print(f"tensor_B.T transposed {tensor_B.T}")
# zatem
print("tensor_A @ tensor_B.T:")
print(torch.matmul(tensor_A, tensor_B.T))
# zwróci popawny wynik dla mnożenia macierzy (2, 3) i (3, 2)

# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(
    f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n"
)
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")
print("tensor_A.T @ tensor_B:")
print(torch.matmul(tensor_A.T, tensor_B))
