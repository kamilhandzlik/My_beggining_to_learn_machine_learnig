# Putting it all together with multiclassification problem
"""
Binary classification = one thing or another (e.g. cat or dog, spam or not spam, etc.)
Multiclass classification = mutliple things (e.g. cat, dog, rabbit, etc.)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper_function import plot_predictions, plot_decision_boundary


NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi class data
X_blob, y_blob = make_blobs(
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1.5,
    random_state=RANDOM_SEED,
)
# 2. Convert to tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

# 3. Split the data into training and testing sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED,
)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 4. Create a dataset and dataloader

# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.title("Blobs Dataset")
# plt.show()

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 5. Vreate model


class MultiClassClassifictionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack_1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack_1(x)


# 6. Instantiate the model
model_4 = MultiClassClassifictionModel(
    input_features=2, output_features=4, hidden_units=8
).to(device)

# 7. Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)


# 8. Create a training loop
torch.manual_seed(RANDOM_SEED)
epochs = 200
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

"""
In order to evaluate and train test our model, we need to convert our model's output logits to predictions
and then to prediction labels. 
Logits (raw output of a model) -> Prediction probabilities (take the argmax of the prediction probabilities)
"""
y_logits = model_4(X_blob_train[:5])
y_pred_probs = torch.softmax(y_logits, dim=1)
# Converting model's prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)


for epoch in range(epochs):
    model_4.train()
    # 1. Forward pass
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    # 2. Calcualte loss
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # 4. loss back propagation
    loss.backward()
    # 5. Optimizer step
    optimizer.step()

    with torch.inference_mode():
        model_4.eval()
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, test_pred)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f} Test accuracy: {test_acc:.2f}%"
        )


# 9. Evaluate the model
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)


print(f"First 10 predictions: {y_logits[:10]}")
y_preds = torch.argmax(y_logits, dim=1)
print(f"Pred labels: {y_preds[:10]}")


""" Odkomentuj żeby zobaczyć wykres"""
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(
    model=model_4,
    X=X_blob_train,
    y=y_blob_train,
)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(
    model=model_4,
    X=X_blob_test,
    y=y_blob_test,
)
plt.show()


# plot_blobs_predictions(X_blob, y_blob, model_4)
