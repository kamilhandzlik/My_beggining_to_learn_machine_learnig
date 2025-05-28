# Pytorch computer vision
# CNN - Convolutional Neural Network
"""
CNN to specjalny rodzaj sztucznej sieci neuronowej, który został zaprojektowany z myślą o przetwarzaniu danych o strukturze siatki,
 takich jak obrazy (dwuwymiarowe siatki pikseli). Ich głównym celem jest automatyczne wykrywanie istotnych cech w danych wejściowych,
   bez konieczności ręcznego projektowania tzw. feature extractorów.

Dzięki mechanizmowi splotu (convolution), CNN potrafi:

wykrywać krawędzie, kształty i tekstury na wczesnych warstwach,

rozpoznawać bardziej złożone wzory (np. twarz, samochód, litera) w warstwach głębszych.

Innymi słowy, CNN to mistrz wzrokowy w świecie algorytmów - Sherlock Holmes analizujący każdy piksel jakby był wskazówką w zagadce.
"""

# Architektura CNN
"""
Tradycyjna architektura CNN składa się z kilku klasycznych komponentów, które - jak to w porządnej orkiestrze - mają swoje role i rytm działania:

1. Warstwa wejściowa (Input Layer)
Przyjmuje dane - np. obraz 32x32 piksele z 3 kanałami (RGB).

Reprezentacja: 32 x 32 x 3

2. Warstwy splotowe (Convolutional Layers)
Główna atrakcja CNN.

Nakładają filtry (jądra) na obraz, przesuwając się po nim i tworząc mapy cech (feature maps).

Każdy filtr wykrywa inny wzorzec: poziome krawędzie, okręgi, faktury itd.

➡️ Reprezentacja przekształca się: 32 x 32 x 3 → 28 x 28 x 16, 24 x 24 x 32 itd. (zależnie od rozmiaru filtra i kroku stride).

3. Funkcja aktywacji (najczęściej ReLU)
Wprowadza nieliniowość (żeby nie skończyć z czymś, co przypomina tylko liniową regresję).

Pomaga modelowi lepiej dopasować się do skomplikowanych danych.

4. Warstwy spłaszczające (Pooling Layers - np. MaxPooling)
Redukują wymiary map cech (czyli rozmiar danych), ale zachowują najważniejsze informacje.

Działają jak starannie przemyślany minimalizm: mniej danych, ale więcej sensu.

5. Warstwa spłaszczająca (Flatten)
Przekształca dane z formatu przestrzennego (np. 6 x 6 x 64) do płaskiego wektora (np. 2304 elementy).

Gotowe do przetworzenia przez klasyczne sieci neuronowe.

6. Warstwy w pełni połączone (Fully Connected - Dense Layers)
Typowa perceptronowa sieć neuronowa na końcu.

Odpowiada za klasyfikację lub regresję - np. rozpoznanie, że na obrazku jest pies, kot lub tost (jeśli CNN miał zły dzień 😉).

7. Warstwa wyjściowa (Output Layer)
Zawiera tyle neuronów, ile klas chcemy rozpoznać (np. 10 w przypadku CIFAR-10).

Zwykle zakończona funkcją softmax (w klasyfikacji wieloklasowej), która przekształca wyniki w prawdopodobieństwa.

Przykład Architektury CNN (dla obrazu 32x32x3)
text
Kopiuj
Edytuj
Input (32x32x3)
↓
Conv2D (filter=3x3, depth=32) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (filter=3x3, depth=64) + ReLU
↓
MaxPooling2D (2x2)
↓
Flatten
↓
Dense (128) + ReLU
↓
Dense (10) + Softmax → Output



Wążne biblioteki, które powinieneś poznać:
- torchvision  - baza z której korzystasz
- torchvision.dataset - ładowanie i tworzenie danych
- torchvision.models - używanie już wytrenowanych modeli
- torchvision.trandforms - dostosowywanie obrazów tak, żeby model mógł je czytać
- torch.utils.data.Dataset
- torch.utils.data.DataLoader
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 1. Przygotowanie danych
# Dataset, którego użyje to FashionMNIST - ekwiwalent do Hello World! XD

# 1.1 Pobieranie Danych z fashionMNIST do folderu data
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None,
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)

RANDOM_SEED = 42
# Sprawdzenie czy jest podział
print(f"len Train data: {len(train_data)}, len test data: {len(test_data)}")

# Sprawdzenie w jakiej formie są zapisane obrazy
image, label = train_data[0]
# print(f"Image: {train_data[0]}")
# Sprawdzenie jakie mamy klasy
class_names = train_data.classes
class_to_idx = train_data.class_to_idx

# print(class_names)
# print(class_to_idx)
# print(train_data.targets)

print(f"Image shape: {image.shape} [Colorchannels, height, witht]")
print(f"Label shape: {class_names[label]}")

# Tak z ciekawości chcę zobaczyć jak mók image wygląda w wykresie matplotlib
# Pamięstaj, że image ma wymiary [kolor, szerokość, wysokość] matplotlib kolor ma na końcu więc musisz zmienić Tensor żeby nie mieć shape errora

## odkomentuj żeby zobaczyć obrazek
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

## odkomentuj żeby zobaczyć obrazek
torch.manual_seed(RANDOM_SEED)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()

## Prepare DataLoader
"""
W tej chwili dane są w formie Pytorch Dataset i trzeba przerobić je na dane iterowalne przez pythona.

Now we've got a dataset ready to go.

The next step is to prepare it with a torch.utils.data.DataLoader or DataLoader for short.

The DataLoader does what you think it might do.

It helps load data into a model.

For training and for inference.

It turns a large Dataset into a Python iterable of smaller chunks.

These smaller chunks are called batches or mini-batches and can be set by the batch_size parameter.

Why do this?

Because it's more computationally efficient.

In an ideal world you could do the forward pass and backward pass across all of your data at once.

But once you start using really large datasets, unless you've got infinite computing power, it's easier to break them up into batches.

It also gives your model more opportunities to improve.

With mini-batches (small portions of the data), gradient descent is performed more often per epoch (once per mini-batch rather than once per epoch).

What's a good batch size?

32 is a good place to start for a fair amount of problems.

But since this is a value you can set (a hyperparameter) you can try all different kinds of values,
 though generally powers of 2 are used most often (e.g. 32, 64, 128, 256, 512).
"""

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Odkomentuj żeby zobaczyć datalodery jak wyglądają, rozmiary próbek
# print(f"Dataloaders: {train_dataloader, test_dataloader}")
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")


# Sprawdzenie co jest w środku dataloaderó:
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

# Show Samples
# torch.manual_seed(RANDOM_SEED)
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, labels = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
# print(f"Image size: {img.shape}")
# print(f"Label: {label}, label size: {label.shape}")
