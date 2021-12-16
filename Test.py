import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import tree
import graphviz
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Utilities import CustomDataset, NN, accuracy

test_percent = 0.2

# Hyper parameters NN
batch_size = 128
learning_rate = 1.5e-4
weight_decay = 1e-4
num_epochs = 160
dropout_percentage = 0.5
num_hidden_units = 20

# Hyper parameters for random forest
max_depth = 4

# Load and shuffle data (with same seed as for validation)
data = pd.read_csv("Data/one_out_of_k_outliers_removed.csv", delimiter=',')
data = data.sample(frac=1, random_state=3895).reset_index(drop=True)  # shuffle rows to produce randoms sets

# Split data into inputs and targets
targets = torch.tensor(
    data["HeartDisease"]
).type(torch.FloatTensor)
inputs = torch.tensor(
    data.drop("HeartDisease", axis=1).values
).type(torch.FloatTensor)
labels = data.drop("HeartDisease", axis=1).columns.tolist()

# Assert everything is as expected
assert len(targets) == inputs.shape[0]
assert len(labels) == inputs.shape[1]

# Split inputs and targets into train and test
split_index = int(len(targets) * test_percent)
X_train = inputs[split_index:]
y_train = targets[split_index:]
X_test = inputs[0:split_index]
y_test = targets[0:split_index]

# Normalize the data
train_means = torch.mean(X_train, axis=0)
train_stds = torch.std(X_train, axis=0)
X_train = (X_train - train_means) / train_stds
X_test = (X_test - train_means) / train_stds

# Train random forest
model = RandomForestClassifier(max_depth=max_depth)
model.fit(X_train, y_train)

# Test model
train_accuracy_RF = accuracy(model.predict(X_train), y_train.numpy())
test_accuracy_RF = accuracy(model.predict(X_test), y_test.numpy())

# Save a random tree from the forest
estimator = model.estimators_[0]

dot_data = tree.export_graphviz(
    estimator,
    out_file=None,
    feature_names=labels,
    class_names=["Negative", "Positive"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("rf")

# Wrap training data in dataloader
DS_train = CustomDataset(X_train, y_train)
DL_train = DataLoader(DS_train, batch_size=batch_size, shuffle=True)

# Define net
net = NN(num_hidden_units, dropout_percentage)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the network
train_losses, test_losses, train_accuracies, test_accuracies, eval_counter = [], [], [], [], []
test_output, train_output = None, None
for epoch in range(num_epochs):
    for index, data in enumerate(DL_train, 0):
        input = data[0]
        target = data[1]

        net.train()
        optimizer.zero_grad()
        output = torch.squeeze(net(input))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate every x epochs
    if epoch % 1 == 0:
        net.eval()
        eval_counter.append(epoch)

        # Evaluate on training data
        train_output = torch.squeeze(net(X_train))
        train_losses.append(
            criterion(train_output, y_train).detach().numpy()
        )
        train_accuracies.append(
            accuracy(
                [1 if x[0] > 0.5 else 0 for x in net(X_train).detach().numpy()],
                y_train.numpy()
            )
        )

        # Evaluate on test data
        test_output = torch.squeeze(net(X_test))
        test_losses.append(
            criterion(test_output, y_test).detach().numpy()
        )
        test_accuracies.append(
            accuracy(
                [1 if x[0] > 0.5 else 0 for x in net(X_test).detach().numpy()],
                y_test.numpy()
            )
        )

# Plot final results
plt.plot(eval_counter, train_losses)
plt.plot(eval_counter, test_losses)
plt.title("Loss on test data during training")
plt.legend(["Train loss", "Test loss"])
plt.show()

plt.plot(eval_counter, train_accuracies)
plt.plot(eval_counter, test_accuracies)
plt.title("Accuracy on test data during training")
plt.legend(["Train accuracy", "Test accuracy"])
plt.show()

# Print final results
print(f"\nNN performance")
print(f"Train loss: {train_losses[-1]}")
print(f"Test loss: {test_losses[-1]}")
print(f"Train accuracy: {train_accuracies[-1]}")
print(f"Test accuracy: {test_accuracies[-1]}")

print(f"\nRF performance")
print(f"Train accuracy: {train_accuracy_RF}")
print(f"Test accuracy: {test_accuracy_RF}")

print(f"\nBase performance")
print(f"Train loss: {nn.BCELoss()(torch.zeros(len(X_train)), y_train).detach().numpy()}")
print(f"Test loss: {nn.BCELoss()(torch.zeros(len(X_test)), y_test).detach().numpy()}")
print(f"Train accuracy: {accuracy(np.zeros(len(X_train)), y_train.numpy())}")
print(f"Test accuracy: {accuracy(np.zeros(len(X_test)), y_test.numpy())}")

# Save loss/accuracy for each observation in order to calculate confidence interval later
np.savetxt(
    "train_losses_NN.csv",
    np.array(
        [
            nn.BCELoss()(output, target).detach().numpy()
            for output, target
            in zip(torch.squeeze(net(X_train)), y_train)
        ]
    ),
    delimiter=","
)

np.savetxt(
    "test_losses_NN.csv",
    np.array(
        [
            nn.BCELoss()(output, target).detach().numpy()
            for output, target
            in zip(torch.squeeze(net(X_test)), y_test)
        ]
    ),
    delimiter=","
)

np.savetxt(
    "train_accuracy_NN.csv",
    [1 if x[0] > 0.5 else 0 for x in net(X_train).detach().numpy()] == y_train.numpy(),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "test_accuracy_NN.csv",
    [1 if x[0] > 0.5 else 0 for x in net(X_test).detach().numpy()] == y_test.numpy(),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "train_accuracy_RF.csv",
    model.predict(X_train) == y_train.numpy(),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "test_accuracy_RF.csv",
    model.predict(X_test) == y_test.numpy(),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "train_loss_base.csv",
    np.array(
        [
            nn.BCELoss()(output, target).detach().numpy()
            for output, target
            in zip(torch.zeros(len(X_train)), y_train)
        ]
    ),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "test_loss_base.csv",
    np.array(
        [
            nn.BCELoss()(output, target).detach().numpy()
            for output, target
            in zip(torch.zeros(len(X_test)), y_test)
        ]
    ),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "train_accuracy_base.csv",
    np.zeros(len(X_train)) == y_train.numpy(),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "test_accuracy_base.csv",
    np.zeros(len(X_test)) == y_test.numpy(),
    delimiter=",",
    fmt="%d"
)