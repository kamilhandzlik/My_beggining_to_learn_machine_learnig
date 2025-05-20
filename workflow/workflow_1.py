"""
Workflow 1: Basic Workflow
1. Get data ready (turn it into a tensors),
2. Build or pick a pre-trained model,
2.1 Pick a loss funstion & optimizer,
2.2 Build a training loop,
3. fit the model to the data and make a prediction,
4. Evaluate the model,
5. Improve throuh experimentation,
6. Save and reload your trained model,
----------------------------------------------------------------

Exercise belov covers:
1. data (prepare and load),
2. build model
3. train model (fitting model to data),
4. making predictions and evaluate model (inference),
5. save and load model,
6. put it all together in a workflow,
-----------------------------------------------------------------

It does not cover:
1. Improve throuh experimentation,
"""

# https://matplotlib.org/
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# Chaking version of pytorch
print(torch.__version__)  # at least 1.10 and cuda

# I.

# Data preparing and loading
"""
Data can be almost anything... in machine learnig.
Example:
- images, text, audio, video,
- tabular data (data in a table), excel files, csv files,
- time series data (data over time),
- DNA, protein sequences,
- Text files, JSON files,
- etc.

Uczenie maszynowe skada się z dwoch części:
1. Zamiana danych na numeryczną reprezentację (tensors),
2. Uczenie modelu na tych danych i tworzenie z nich wzorów. 

Zaczynam z tworzeniem znanych danych za pomocą regresji liniowej (linear regression).
"""

# Tworzenie linii prostej za pomocą regresji liniowej z znanymi współczynnikami
# y = mx + b

# Współczynniki
weight = 0.7  # weight=m
bias = 0.3  # bias=b

# Tworzenie danych
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(
    f"X: {X[:10]} \n y: {y[:10]},\nlen(y): {len(y)}, len(x): {len(X)}"
)  # first 10 values of X and y


"""
 Spliting data into train and test sets (one of the most important steps in ML)
 Training set is used to train the model, 60-80% used always
 Test set is used to test the model, 10-20% used always
 Validation set is used to validate the model, 10-20% used often but not always
"""
# Create train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Visualization of data using matplotlib


def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    """
    Plot training data, test data and compare predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Tesing data")

    # Are there any predictions?
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    # Wywołanie wykresu
    plt.show()


# print(plot_predictions())


# II. Build model
"""
My first model :)
It will be a linear regression model

What his model does:
1. Start with random values for weights and bias,
2. Look at training data and adjust the random values to better represent (or get closer to)
   the ideal values (the weight & bias, valuse we used to create the data)

How does it do what it does?
1. Gradient descent
2. Backpropagation
"""


# Create a linear regression model class
class LiniearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # Forward method to difine computation in the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x- is the input data
        return self.weights * x + self.bias  # linera regresion data


"""
PyTorch model building essentials:
1. nn.Module -    contains the larger building blocks (layers),

2. nn.Parameter - contains the smaller parameters like weights and biases
                  (put these together to make nn.Module(s)),

3. forward() -   tells the larger blocks how to make calculations on inputs
                 (tensors full of data) within nn.Module(s),

4. torch.optim - contains optimization methods on how to improve the parameters
                   within nn.Parameter to better represent input data,
"""

# Creating random seed
torch.manual_seed(42)

# Create a model instance (subclass od nn.Module)
model_0 = LiniearRegressionModel()
print(list(model_0.parameters()))


# List named parameters of the model
print(model_0.state_dict())


# Przewidywanie modelu używając torch.inference_mode()

with torch.inference_mode():
    y_preds = model_0(X_test)
    print(y_preds)
    print(y_test)
    # print(plot_predictions(predictions=y_preds))
    print(y_preds == y_test)  # porównanie przewidywań z danymi testowymi
    print(y_preds[:10], y_test[:10])  # first 10 values of y_preds and y_test


# III. Train model
"""
Cała idea trenowania modelu polega na tym żeby przeszedł z przypadkowych parametrów
(współczynników) do współczynników które lepiej pasują do danych i tak w kółko do czasu aż 
model nie zacznie zachowywać się tak jak chcemy.

Jednym ze sposobów na weryfikację tego jak w danej chwili twój model się zachowuje jest loss function. 

Creating a loss function and optimizer in PyTorch
For our model to update its parameters on its own, we'll need to add a few more things to our recipe.

And that's a loss function as well as an optimizer.

The rolls of these are:

Function	What does it do?	Where does it live in PyTorch?	Common values
Loss function	Measures how wrong your model's predictions (e.g. y_preds) are compared to the truth labels (e.g. y_test). Lower the better.
PyTorch has plenty of built-in loss functions in torch.nn.
Mean absolute error (MAE) for regression problems (torch.nn.L1Loss()).
Binary cross entropy for binary classification problems (torch.nn.BCELoss()).
Optimizer	Tells your model how to update its internal parameters to best lower the loss.
You can find various optimization function implementations in torch.optim.
Stochastic gradient descent (torch.optim.SGD()). Adam optimizer (torch.optim.Adam()).
Let's create a loss function and an optimizer we can use to help improve our model.

Depending on what kind of problem you're working on will depend on what loss function and what optimizer you use.

However, there are some common values, that are known to work well such as the SGD (stochastic gradient descent) or Adam optimizer.
And the MAE (mean absolute error) loss function for regression problems (predicting a number) or binary cross entropy loss function for
classification problems (predicting one thing or another).

For our problem, since we're predicting a number, let's use MAE (which is under torch.nn.L1Loss()) in PyTorch as our loss function.
"""

# Setup loss function
loss_fn = nn.L1Loss()

# Setup optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)  # lr = learnig rate


# print(loss_fn)
# print(optimizer)


# Building a training loop and testing loop
# Parę rzeczy potrzebnych do stworzenia tych pętli:

"""
0. Loop through the data
1. Forward pass/forward propagation (model prediction)
2. Calculate the loss (how wrong the model is)
3. Zero gradients/Optimizer zero grad (clear old gradients)
4. Backward pass/loss backward (calculate the gradients) (backpropagation)
5. Step the optimizer (update the model parameters) (gradient descent)
6. Testing loop (optional)
"""

# An epoch is one loop through the data
epochs = 200  # number of epochs

epoch_count = []
loss_values = []
test_loss_values = []

# Testowanie modelu

with torch.inference_mode():
    # with torch.no_grad(): # Przyspiesza obliczenia
    test_pred = model_0(X_test)
    y_preds_new = model_0(X_test)
    # print(y_preds_new)
    # print(plot_predictions(predictions=y_preds_new))

# def train_step(model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
for epoch in range(epochs):
    # set model to training mode
    model_0.train()

    # 1. Forward pass
    # 1. Forward pass/forward propagation (model prediction)
    y_preds = model_0(X_train)

    # 2. Obliczenie błędu
    # 2. Calculate the loss (how wrong the model is)
    loss = loss_fn(y_preds, y_train)

    # 3. Wyzerowanie gradientów
    # 3. Zero gradients/Optimizer zero grad (clear old gradients)
    optimizer.zero_grad()

    # 4. Backpropagation
    # 4. Backward pass/loss backward (calculate the gradients)
    loss.backward()

    # 5. Aktualizacja wag
    # 5. Step the optimizer (update the model parameters) (gradient descent)
    optimizer.step()

    # Co 10 epok pokaż co się dzieje
    if epoch % 10 == 0:
        # Oblicz test loss na aktualnych wagach modelu
        model_0.eval()
        with torch.inference_mode():
            test_pred = model_0(X_test)
            test_loss = loss_fn(test_pred, y_test)
        epoch_count.append(epoch)
        loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(
            f"Epoch {epoch} | Loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f}"
        )

# print(train_step(model_0, loss_fn, optimizer))  # train the model
# print(model_0.state_dict())  # print model parameters


# with torch.inference_mode():
#     # with torch.no_grad(): # Przyspiesza obliczenia
#     test_pred = model_0(X_test)
#     test_loss = loss_fn(test_pred, y_test)

plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
print(f"Epoch: {epoch} | Test: {loss} | Test loss: {test_loss}")  # print test loss


# Saving model in pytorch
"""
torch.save(model_0.state_dict(), "model_0.pth")  # save model parameters
torch.save(model_0, "model_0.pth")  # save model
tprch.load("model_0.pth")  # load model
model_0.load_state_dict(torch.load("model_0.pth"))  # load model parameters
torch.nn.Module.load_state_dict(torch.load("model_0.pth"))  # this allows to load a model's saved dictionary
"""

from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)


# 4. Load the model
#  To load first we need to make new instance of the model
loaded_model_0 = LiniearRegressionModel()

# 5. Load the state_dict of model_0
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))


# Evaluate loaded model
loaded_model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
    loaded_model_1_preds = loaded_model_0(X_test)
print(torch.allclose(y_preds, loaded_model_1_preds))
