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


print(plot_predictions())


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
