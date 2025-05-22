"""
Exercises
All of the exercises are focused on practicing the code above.

You should be able to complete them by referencing each section or by following the resource(s) linked.

Resources:

Exercise template notebook for 00.
Example solutions notebook for 00 (try the exercises before looking at this).
1. Documentation reading - A big part of deep learning (and learning to code in general) is getting familiar with the documentation of a certain
   framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes
   reading the following (it's okay if you don't get some things for now, the focus is not yet full understanding, it's awareness).
   See the documentation on torch.Tensor and for torch.cuda.
2. Create a random tensor with shape (7, 7).
3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
4. Set the random seed to 0 and do exercises 2 & 3 over again.
5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation
   for torch.cuda for this one). If there is, set the GPU random seed to 1234.
6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when
   creating the tensors (this doesn't have to be the GPU random seed).
7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
8. Find the maximum and minimum values of the output of 7.
9. Find the maximum and minimum index values of the output of 7.
10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10).
    Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.

Extra-curriculum
Spend 1-hour going through the PyTorch basics tutorial.
https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
To learn more on how a tensor can represent data, see this video: What's a tensor? https://www.youtube.com/watch?v=f5liqUk0ZTw
"""

import torch

# 2. Create a random tensor with shape (7, 7).
random_tensor = torch.rand(7, 7)
print(f"\033[32mExercise 2\033[0m\n{random_tensor, random_tensor.shape}\n")


# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)
#   (hint: you may have to transpose the second tensor).
random_tensor_2 = torch.rand(1, 7)
print(f"\033[32mExercise 3\033[0m\n{random_tensor @ random_tensor_2.T}\n")

# 4. Set the random seed to 0 and do exercises 2 & 3 over again.
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
random_tensor = torch.rand(7, 7)
print(f"\033[32mExercise 4\033[0m\n{random_tensor}\n")
torch.manual_seed(RANDOM_SEED)
random_tensor_2 = torch.rand(1, 7)
print(f"\033[32mExercise 4\033[0m\n{random_tensor @ random_tensor_2.T}\n")

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation
#    for torch.cuda for this one). If there is, set the GPU random seed to 1234.
RANDOM_GPU_SEED = 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_GPU_SEED)
    print(f"\033[32mExercise 5\033[0m\nGPU random seed set to {RANDOM_GPU_SEED}\n")


# 6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when
#    creating the tensors (this doesn't have to be the GPU random seed).
if torch.cuda.is_available():
    torch.manual_seed(1234)
    random_tensor_ex6_1 = torch.rand(2, 3).cuda()
    torch.manual_seed(1234)
    random_tensor_ex6_2 = torch.rand(2, 3).cuda()
    print(f"\033[32mExercise 6\033[0m\n{random_tensor_ex6_1}\n{random_tensor_ex6_2}\n")

# 7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
if torch.cuda.is_available():
    print(f"\033[32mExercise 6\033[0m\n{random_tensor_ex6_1 @ random_tensor_ex6_2.T}\n")

# 8. Find the maximum and minimum values of the output of 7.
if torch.cuda.is_available():
    output_ex7 = random_tensor_ex6_1 @ random_tensor_ex6_2.T
    print(
        f"\033[32mExercise 8\033[0m\n Max: {torch.max(output_ex7)}\n Min: {torch.min(output_ex7)}\n"
    )

# 9. Find the maximum and minimum index values of the output of 7.
if torch.cuda.is_available():
    print(
        f"\033[32mExercise 9\033[0m\nMax index: {output_ex7.argmax()}\nMin index: {output_ex7.argmin()}\n"
    )

# 10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10).
# Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
RANDOM_SEED_EX_10 = 7
torch.manual_seed(RANDOM_SEED_EX_10)
random_tensor_ex10 = torch.rand(1, 1, 1, 10)
torch.manual_seed(RANDOM_SEED_EX_10)
random_tensor_ex10_reshaped = random_tensor_ex10.squeeze()

print(
    f"""\033[32mExercise 10\033[0m\nRandom tensor 1: {random_tensor_ex10}\nShape 1: {random_tensor_ex10.shape}\n 
Random tensor 2: {random_tensor_ex10_reshaped}\nShape 2: {random_tensor_ex10_reshaped.shape}\n"""
)


# Testy używać za pomocą pytesta (komenda pytest nazwapliku.py -v)
def test_exercise_2():
    tensor = torch.rand(7, 7)
    assert tensor.shape == (7, 7), "Tensor should have shape (7, 7)"


def test_exercise_3():
    a = torch.rand(7, 7)
    b = torch.rand(1, 7)
    result = a @ b.T
    assert result.shape == (7, 1), f"Result shape should be (7, 1), got {result.shape}"


def test_exercise_4():
    torch.manual_seed(0)
    t1 = torch.rand(7, 7)
    torch.manual_seed(0)
    t2 = torch.rand(1, 7)
    torch.manual_seed(0)
    t3 = torch.rand(7, 7)
    torch.manual_seed(0)
    t4 = torch.rand(1, 7)
    assert torch.equal(t1, t3), "Tensors with the same seed should be equal"
    assert torch.equal(t2, t4), "Second tensors with same seed should also match"


def test_exercise_5():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        # Nie da się łatwo sprawdzić deterministycznego zachowania GPU bez dodatkowych operacji,
        # więc traktujemy to jako smoke test — jeśli nie wywali wyjątku, jest OK.
        assert True


def test_exercise_6_and_7():
    if torch.cuda.is_available():
        torch.manual_seed(1234)
        t1 = torch.rand(2, 3).cuda()
        torch.manual_seed(1234)
        t2 = torch.rand(2, 3).cuda()
        assert torch.equal(t1, t2), "Tensors with same seed should be equal"
        result = t1 @ t2.T
        assert result.shape == (
            2,
            2,
        ), f"Multiplication result shape should be (2, 2), got {result.shape}"


def test_exercise_8():
    if torch.cuda.is_available():
        torch.manual_seed(1234)
        t1 = torch.rand(2, 3).cuda()
        t2 = torch.rand(2, 3).cuda()
        output = t1 @ t2.T
        max_val = torch.max(output)
        min_val = torch.min(output)
        assert max_val >= min_val, "Max value should be >= Min value"


def test_exercise_9():
    if torch.cuda.is_available():
        torch.manual_seed(1234)
        t1 = torch.rand(2, 3).cuda()
        t2 = torch.rand(2, 3).cuda()
        output = t1 @ t2.T
        max_index = output.argmax()
        min_index = output.argmin()
        assert isinstance(max_index.item(), int), "Argmax should return a tensor index"
        assert isinstance(min_index.item(), int), "Argmin should return a tensor index"


def test_exercise_10():
    torch.manual_seed(7)
    t = torch.rand(1, 1, 1, 10)
    t_squeezed = t.squeeze()
    assert t.shape == (1, 1, 1, 10), "Initial tensor should have shape (1,1,1,10)"
    assert t_squeezed.shape == (10,), "Squeezed tensor should have shape (10,)"
