# Same as ex.1 but using linear layer from nn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Wspóczynniki
weight = 0.80
bias = 0.20

# 1. Tworzenie danych
a = 1
b = weight
c = bias
start = 0
end = 1.5
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
X_SHIFTED = X - 3
X_quad = torch.cat([X_SHIFTED, X_SHIFTED**2], dim=1)

y = a * X_SHIFTED**2 + b * X_SHIFTED + c

print(f"x: {X_SHIFTED}\ny: {y}\nlen x: {len(X_SHIFTED)}\nlen y: {len(y)}")


# Create train/test split
train_split = int(0.8 * len(X_SHIFTED))
X_train, y_train = X_quad[:train_split], y[:train_split]
X_test, y_test = X_quad[train_split:], y[train_split:]

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
    plt.scatter(
        train_data[:, 0], train_labels.squeeze(), c="b", s=4, label="Train data"
    )
    plt.scatter(
        test_data[:, 0], test_labels.squeeze(), c="g", s=4, label="Testing data"
    )
    if predictions is not None:
        plt.scatter(
            test_data[:, 0], predictions.squeeze(), c="r", s=4, label="Predictions"
        )
    plt.legend(prop={"size": 14})
    plt.show()


# 2. Tworzenie modelu
class QuadraticFunctionWithShiftV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=2, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_3 = QuadraticFunctionWithShiftV2().to(device)
model_3.to(device)
print(next(model_3.parameters()).device)


# loss function & optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(params=model_3.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 200
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_3.train()
    y_pred = model_3(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        test_prod = model_3(X_test)
        test_loss = loss_fn(test_prod, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}")


model_3.eval()
with torch.inference_mode():
    y_preds = model_3(X_test)
    test_loss = loss_fn(y_preds, y_test)

plot_predictions(predictions=y_preds.cpu())


MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "quadratic_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model_3.state_dict(), MODEL_SAVE_PATH)

# Load the model
loaded_model = QuadraticFunctionWithShiftV2()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
