import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from torch.utils.data import DataLoader
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier

from Utilities import CustomDataset, NN, accuracy

test_percent = 0.2
num_folds = 5

# Hyper parameters NN
batch_size = 128
learning_rate = 1.5e-4
weight_decay = 1e-4
num_epochs = 600
dropout_percentage = 0.5
num_hidden_units = [5, 10, 20, 40, 80]
train_losses_NN = np.zeros((num_folds, len(num_hidden_units)))
train_accuracies_NN = np.zeros((num_folds, len(num_hidden_units)))
valid_losses_NN = np.zeros((num_folds, len(num_hidden_units)))
valid_accuracies_NN = np.zeros((num_folds, len(num_hidden_units)))

# Hyper parameters for random forest
max_depth = [1, 2, 3, 4, 5]
train_accuracies_RF = np.zeros((num_folds, len(max_depth)))
valid_accuracies_RF = np.zeros((num_folds, len(max_depth)))

# Load and shuffle data
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

# Use cross validation on train data to evaluate model performance
cross_validation = model_selection.KFold(n_splits=num_folds, shuffle=True)

fold_counter = 0
for train_index, validation_index in cross_validation.split(X_train):
    # Extract data for current fold
    X_train_CV = X_train[train_index, :]
    y_train_CV = y_train[train_index]
    X_valid_CV = X_train[validation_index, :]
    y_valid_CV = y_train[validation_index]

    # Normalize data for current fold
    train_means_CV = torch.mean(X_train_CV, axis=0)
    train_stds_CV = torch.std(X_train_CV, axis=0)
    X_train_CV = (X_train_CV - train_means_CV) / train_stds_CV
    X_valid_CV = (X_valid_CV - train_means_CV) / train_stds_CV

    # Wrap training data in dataloader
    DS_train = CustomDataset(X_train_CV, y_train_CV)
    DL_train = DataLoader(DS_train, batch_size=batch_size, shuffle=True)

    # Validate min purity gain for RF
    for i in range(len(max_depth)):
        # Train random forest
        model = RandomForestClassifier(max_depth=max_depth[i])
        model.fit(X_train_CV, y_train_CV)

        # Save performance
        train_accuracies_RF[fold_counter, i] = \
            accuracy(model.predict(X_train_CV), y_train_CV.numpy())
        valid_accuracies_RF[fold_counter, i] = \
            accuracy(model.predict(X_valid_CV), y_valid_CV.numpy())

    # Validate number of hidden units for NN
    for i in range(len(num_hidden_units)):
        # Define net
        net = NN(num_hidden_units[i], dropout_percentage)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train the network
        train_losses, valid_losses, eval_counter = [], [], []
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
                train_output = torch.squeeze(net(X_train_CV))
                train_losses.append(
                    criterion(train_output, y_train_CV).detach().numpy()
                )

                # Evaluate on validation data
                valid_output = torch.squeeze(net(X_valid_CV))
                valid_losses.append(
                    criterion(valid_output, y_valid_CV).detach().numpy()
                )

        # Save final performance
        train_losses_NN[fold_counter, i] = train_losses[-1]
        valid_losses_NN[fold_counter, i] = valid_losses[-1]

        train_accuracies_NN[fold_counter, i] = \
            accuracy(
                [1 if x[0] > 0.5 else 0 for x in net(X_train_CV).detach().numpy()],
                y_train_CV.numpy()
            )

        valid_accuracies_NN[fold_counter, i] = \
            accuracy(
                [1 if x[0] > 0.5 else 0 for x in net(X_valid_CV).detach().numpy()],
                y_valid_CV.numpy()
            )

        # Plot results
        plt.plot(eval_counter, train_losses, label="Training loss")
        plt.plot(eval_counter, valid_losses, label="Validation loss")
        plt.title(f"Fold {fold_counter + 1}, hidden units {num_hidden_units[i]}")
        plt.legend(["Train loss", "Validation loss"])
        plt.show()

    fold_counter += 1

# Calculate averages
train_losses_NN = np.round(np.vstack((
    train_losses_NN,
    np.mean(train_losses_NN, axis=0)
)), 3)

valid_losses_NN = np.round(np.vstack((
    valid_losses_NN,
    np.mean(valid_losses_NN, axis=0)
)), 3)

train_accuracies_NN = np.round(np.vstack((
    train_accuracies_NN,
    np.mean(train_accuracies_NN, axis=0)
)), 3)

valid_accuracies_NN = np.round(np.vstack((
    valid_accuracies_NN,
    np.mean(valid_accuracies_NN, axis=0)
)), 3)

train_accuracies_RF = np.round(np.vstack((
    train_accuracies_RF,
    np.mean(train_accuracies_RF, axis=0)
)), 3)

valid_accuracies_RF = np.round(np.vstack((
    valid_accuracies_RF,
    np.mean(valid_accuracies_RF, axis=0)
)), 3)

# Display final results
print("\nTraining losses for NN")
print(tabulate(
    train_losses_NN,
    headers=num_hidden_units,
))

print("\nValidation losses NN")
print(tabulate(
    valid_losses_NN,
    headers=num_hidden_units,
))

print("\nTraining accuracies for NN")
print(tabulate(
    train_accuracies_NN,
    headers=num_hidden_units
))

print("\nValidation accuracies for NN")
print(tabulate(
    valid_accuracies_NN,
    headers=num_hidden_units
))

print("\nTraining accuracies for RF")
print(tabulate(
    train_accuracies_RF,
    headers=max_depth,
))

print("\nValidation accuracies for RF")
print(tabulate(
    valid_accuracies_RF,
    headers=max_depth,
))

# Save results as csv
np.savetxt("train_losses_NN.csv", train_losses_NN, delimiter=",", fmt="%.3f")
np.savetxt("valid_losses_NN.csv", valid_losses_NN, delimiter=",", fmt="%.3f")
np.savetxt("train_accuracies_NN.csv", train_accuracies_NN, delimiter=",", fmt="%.3f")
np.savetxt("valid_accuracies_NN.csv", valid_accuracies_NN, delimiter=",", fmt="%.3f")
np.savetxt("train_accuracies_RF.csv", train_accuracies_RF, delimiter=",", fmt="%.3f")
np.savetxt("valid_accuracies_RF.csv", valid_accuracies_RF, delimiter=",", fmt="%.3f")
