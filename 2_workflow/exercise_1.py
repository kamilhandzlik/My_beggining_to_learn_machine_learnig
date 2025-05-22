# putting it all together eg. workflow 1 put together

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Wspóczynniki
weight = 0.75
bias = 0.25

# 1. Tworzenie danych
# Tutaj funkcji kwadratowej y =ax^2 +bx + c i wektorze przesunięcie o 2 w prawo wektor [p=2, q=0]
a = 1
b = weight
c = bias
start = -1
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
X_SHIFTED = X - 2  # przesunięcie o 2 w prawo wektor [p=2, q=0]
y = a * X_SHIFTED**2 + b * X_SHIFTED + c

print(f"x: {X_SHIFTED}\ny: {y}\nlen x: {len(X_SHIFTED)}\nlen y: {len(y)}")

# Create train/test split
train_split = int(0.8 * len(X_SHIFTED))
X_train, y_train = X_SHIFTED[:train_split], y[:train_split]
X_test, y_test = X_SHIFTED[train_split:], y[train_split:]

print(
    f"len X_train: {len(X_train)}\nlen y_train: {len(y_train)}\nlen X_test: {len(X_test)}\nlen y_test: {len(y_test)}"
)


def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Train data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.title("Train and test data", fontsize=20)
    plt.show()


# 2. Tworzenie modelu
class QuadraticFunctionWithShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.weight2 = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight2 * x**2 + self.weights * x + self.bias
        # return a * x**2 + self.weights * x + self.bias


torch.manual_seed(42)
model_1 = QuadraticFunctionWithShift()
print(list(model_1.parameters()))

with torch.inference_mode():
    y_pred = model_1(X_test)
    # print(f"y_pred: {y_pred}")
    # print(f"y_test: {y_test}")
    # print(y_pred == y_test)

# 3. Uczenie modelu
# Setup loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)
epochs = 50
epoch_count = []
loss_values = []
test_loss_values = []

with torch.inference_mode():
    test_pred = model_1(X_test)
    y_preds_new = model_1(X_test)

for epoch in range(epochs):
    # 1. Forward pass
    y_pred = model_1(X_train)
    # 2. Calcuate loss
    loss = loss_fn(y_pred, y_train)
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. Backward pass
    loss.backward()
    # 5. Optimizer step
    optimizer.step()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:

        epoch_count.append(epoch)
        loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

plot_predictions(
    predictions=y_preds_new,
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
)

plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
print(f"Epoch: {epoch} | Train loss: {loss.item()} | Test loss: {test_loss.item()}")
