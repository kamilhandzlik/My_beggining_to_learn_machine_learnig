#########################################################################
# Runnning tensors and pytorch on the GPU (and making faster computing) #
#########################################################################

# GPUs = faster computing on numbers thanks to cuda + nvidia hardware + Pytorch working behind the sceenes

# Getting GPU
# Easiest - google collab for a free gpu


# Sprawdzenie czy można używać GPU
import torch

print(torch.cuda.is_available())

# Jeśli output jest True to można używać swojej karty graficznej


# Setup device agonostic code - good practice
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Count number of devices
print(torch.cuda.device_count())


# https://docs.pytorch.org/docs/stable/notes/cuda.html


# Putting tensors and models on the GPU
# 1. Create a tensor and put it on the GPU
# tensor not in GPU
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)

# Move tensor to gpu if available
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)


# NumPy works only on CPU so sometimies you want to move tensor back to CPU
# Move tensor back to CPU
tensor_back_on_cpu = tensor_on_gpu.to("cpu")
tensor_back_on_cpu = tensor_on_gpu.cpu()  # another way to do it
tensor_back_on_cpu = tensor_on_gpu.detach().cpu()  # another way to do it
tensor_back_on_cpu.numpy()  # convert to numpy array that will not return error
print(tensor_back_on_cpu, tensor_back_on_cpu.device)
