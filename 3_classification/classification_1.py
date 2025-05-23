# Neural network classification with PyTorch
"""
Klasyfikacja jest to problem przewidywania.
Czy coś jest jedną rzeczą? Czy zupenie inną?
(może być wiele klas)
"""
# importy
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_function import plot_predictions, plot_decision_boundary


# 1. Make classification data and get it ready
#  Make 1000 samples

n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)
# print(f"len X: {len(X)} len y: {len(y)}")
# print(f"First 5 samples of X: {X[:5]}")
# print(f"First 5 samples of y: {y[:5]}")


# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
# print(f"First 5 samples of circles:\n{circles.head(10)}")


# Visualize the data
def plot_circles(x=X, y=y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Circles dataset")
    plt.show()


# odkomentuj, aby zobaczyć wykres
# plot_circles()
"""
Note The data we're working with is often reffered to as a "toy dataset"
because it's small and easy to work with.
"""

#  Checking input and output shapes
# print(f"X shape: {X.shape}, y shape: {y.shape}")
# print(f"Values for one sample of X: {X[0]} and the same for y: {y[0]}")
# print(f"Shape for one sample of X: {X.shape} and the same for y: {y.shape}")

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Podzielenie danych na zbiót treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
# print(f"len X_train: {len(X_train)}, len y_train: {len(y_train)}")
# print(f"len X_test: {len(X_test)}, len y_test: {len(y_test)}")


# 2. Create a model
"""
Let's create a model to classify the circles dataset.
To do so we want to:
1. Setup device agnostic code so our code will run on an accelerator (GPU) if available.
2. Create a model by subclassing nn.Module.
3. Define a loss function.
4. Define an optimizer.
5. Create a training and testing loop.
"""

# 1.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2.
"""
1. Subclasses nn.Module (almost all models in Pytorch subclass nn.Module)
2. Create 2 nn.Linear() layers that are capable of taking in 2 features (X1 and X2) and outputting 1 value (0 or 1).
3. Define a forward() method that takes in data and returns a prediction.
4. Instatiate the model and send it to the target device.
"""


# 2.1
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        ## 2.2
        ## bierze 2 cechy (X1 i X2) i zwraca 5 cech
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        ## bierze 5 cech z layer 1 i zwraca 1 cechę
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

        ## Przykład użycia nn.Sequential
        # self.two_linear_layers = nn.Sequential(
        # nn.Linear(in_features=2, out_features=5), nn.Linear(in_features=5, out_features=1)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2.3 Define a forward method that outlines the forward pass
        # x -> layer_1 -> layer_2 -> output i tak w koło macieju jak se zrobisz sto layerów to musisz zapisać je wszystkie uwzględnić
        return self.layer_2(self.layer_1(x))


# 2.4
model_0 = CircleModelV0().to(device)
# print(model)
# print(f"Model device: {next(model.parameters()).device}")


#  Replikowanie modelu powyżej za pomocą nn.Sequential w zasadzie to samo co stworzenie klasy ale z mniejszą ilością kodu
# model_0 = nn.Sequential(
# nn.Linear(in_features=2, out_features=5), nn.Linear(in_features=5, out_features=1)
# ).to(device)

# 3. Define a loss function / make predictions
"""
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Lenght of predictions: {len(untrained_preds)} Shape: {untrained_preds.shape}")
print(f"Lenght of test samples: {len(X_test)} Shape: {X_test.shape}")
print(f"First 10 predictions:\n{untrained_preds[:10]}")
print(f"First 10 test samples:\n{X_test[:10]}")
"""
"""
How to chose a loss function?
1.  Binary classification (2 classes) -> Binary Cross Entropy Loss
2.  Multiclass classification (3+ classes) -> Cross Entropy Loss
3.  Regression (predicting a number) -> Mean Squared Error Loss
4.  Multilabel classification (multiple classes) -> Binary Cross Entropy Loss
5.  Multilabel regression (multiple numbers) -> Mean Squared Error Loss
6.  Image segmentation (pixel-wise classification) -> Cross Entropy Loss
7.  Object detection (bounding box regression) -> Mean Squared Error Loss
8.  Image generation (generating new images) -> Binary Cross Entropy Loss
9.  Image classification (classifying images) -> Cross Entropy Loss
10. Text classification (classifying text) -> Cross Entropy Loss
11. Text generation (generating text) -> Cross Entropy Loss
12. Text summarization (summarizing text) -> Cross Entropy Loss
13. Text translation (translating text) -> Cross Entropy Loss

https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a/
https://www.learnpytorch.io/01_pytorch_workflow/#creating-a-loss-function-and-optimizer-in-pytorch
"""
loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with logits
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 4. Training and testing loop
epochs = 200

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

## View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
# print(y_logits)
## Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)
# print(y_pred_probs)
## Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

## In full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

## Check for equality
# print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

## Get rid of extra dimension
y_preds.squeeze()

for epoch in range(epochs):
    model_0.train()

    # 4.1 Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 4.2 Calculating loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 4.3 Optimizer zero grad
    optimizer.zero_grad()

    # 4.4 Backward pass
    loss.backward()

    # 4.5 Optimizer step
    optimizer.step()

    # Testig
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )

# This model does not seem to learn. Let's evaluate it
if Path("helper_function.py").is_file():
    print("halper_function.py already exists, skipping download")
else:
    print("Donloading helper_function.py")
    requests = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py")
    with open("helper_function.py", 'wb') as f:
        f.write(requests.content)


# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 1)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

# 5. Improving a model (from a model perspective)
# Let's try to fix our model's underfitting problem.
"""
* Add more layers	Each layer potentially increases the learning capabilities of the model with each layer being able to learn some kind of new
  pattern in the data. More layers are often referred to as making your neural network deeper.
* Add more hidden units	Similar to the above, more hidden units per layer means a potential increase in learning capabilities of the model.
* More hidden units are often referred to as making your neural network wider.
* Fitting for longer (more epochs)	Your model might learn more if it had more opportunities to look at the data.
* Changing the activation functions	Some data just can't be fit with only straight lines (like what we've seen), using non-linear activation
  functions can help with this (hint, hint).
* Change the learning rate	Less model specific, but still related, the learning rate of the optimizer decides how much a model should change
  its parameters each step, too much and the model overcorrects, too little and it doesn't learn enough.
* Change the loss function	Again, less model specific but still important, different problems require different loss functions. For example,
  a binary cross entropy loss function won't work with a multi-class classification problem.
* Use transfer learning	Take a pretrained model from a problem domain similar to yours and adjust it to your own problem. We cover transfer
  learning in notebook 06.


  because you can adjust all of these by hand, they're referred to as hyperparameters.

Let;s imporeve model by:
* Adding more hidden layers
* Increase the numer of layers: 2 -> 3
* Increas the number of epochs 100 -> 1000 

  but this willl b in classification_2 to not make too much of a mess here in the code
"""

