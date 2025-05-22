import torch

# Tensor datatypes
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None)
print(f"Float 32 tensor: {float_32_tensor.dtype}")

# float 32 zostanie podstawowo zwrócony między innymi gdy dtype ustawimy na none
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(f"Float 16 tensor: {float_16_tensor.dtype}")


# Link do dostępnych typów danych w Pytorch
# https://docs.pytorch.org/docs/stable/tensors.html

# Dlaczego 32, 16 i 64 mają znaczenie?
# Odpowiedzią jest precyzja, która ma kluczowe znaczenie przy sieciach neuronowych
# mnniejsza przycja oznacza szybsze działanie i mniej zajętego miejsca na dysku większ precyzja działa odwrotnie.
# https://en.wikipedia.org/wiki/Precision_(computer_science)

"""
Tensor datatypes to jeden z 3 wielkich errorów w Pytorchu
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on the right device 
"""

float_32_tensor_2 = torch.tensor(
    [3.0, 6.0, 9.0],
    dtype=None,
    device=None,
    requires_grad=False,
)
print(f"2. Float 32 tensor: {float_32_tensor_2}")

float_16_tensor_2 = float_32_tensor_2.type(torch.float16)  # można też użyć torch.half

# 32-bit complex
tensor_32_bit_complex = torch.tensor([3.0, 6.0, 9.0], dtype=(torch.complex32))
print(f"tensor_32_bit_complex: {tensor_32_bit_complex}")


"""
W PyTorch (i ogólnie w programowaniu), różnica między unsigned (bez znaku) a signed (ze znakiem) integer polega na tym, czy liczby mogą być ujemne:

Unsigned Integer (bez znaku):

Może przechowywać tylko liczby nieujemne (0 i liczby dodatnie).
Cały zakres wartości jest przeznaczony na liczby dodatnie.
Przykład: uint8 (8-bitowy unsigned integer) ma zakres od 0 do 255.
Signed Integer (ze znakiem):

Może przechowywać zarówno liczby ujemne, jak i dodatnie.
Połowa zakresu jest przeznaczona na liczby ujemne, a druga połowa na dodatnie.
Przykład: int8 (8-bitowy signed integer) ma zakres od -128 do 127.
W PyTorch możesz używać tych typów danych w zależności od potrzeb. Na przykład:


"""
# Unsigned integer (uint8)
tensor_uint8 = torch.tensor([0, 255], dtype=torch.uint8)
print(f"Unsigned tensor: {tensor_uint8}")

# Signed integer (int8)
tensor_int8 = torch.tensor([-128, 127], dtype=torch.int8)
print(f"Signed tensor: {tensor_int8}")

# 8-bit integer (unsigned) / Dostępne są uint8, 16, 32, 64
tensor_uint_8 = torch.tensor([3.0, 6.0, 9.0], dtype=(torch.uint8))
print(f"8-bit integer (unsigned): {tensor_uint_8}")
# przykładow po dodania tutaj minusa (tensor_uint_8) do której kolwiek z liczb wyświetli następujący error
"""
Traceback (most recent call last):
  File "e:\My_beggining_to_learn_machine_learnig\fundamentals\tensors_4_datatypes.py", line 69, in <module>
    tensor_uint_8 = torch.tensor([3.0, -6.0, 9.0], dtype=(torch.uint8))       
RuntimeError: value cannot be converted to type uint8 without overflow 
"""

# 8-bit integer (signed) / Dostępne są int8, 16, 32, 64
tensor_int_8 = torch.tensor([-3.0, -6.0, 9.0], dtype=(torch.int8))
print(f"8-bit integer (signed): {tensor_int_8}")


# Diffrent datatypes can operate with ine another althougt there may be conflicts, this two examples actually work.
print(
    f"multiplying tensors with different float datatype: {float_16_tensor_2 * float_32_tensor_2}"
)
print(
    f"multiplying tensors with different datatype float x int: {float_16_tensor_2 * tensor_int_8}"
)


int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int64)
print(int_32_tensor * float_32_tensor)
