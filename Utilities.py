import torch.nn as nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class NN(nn.Module):
    def __init__(self, num_hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(18, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def accuracy(prediction, target):
    return (prediction == target).mean()