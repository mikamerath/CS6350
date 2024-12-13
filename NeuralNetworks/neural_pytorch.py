import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd


# num_features (should be 4)
# num_layers is {3, 5, or 9}
# hidden_layer with is {5, 10, 25, 50, 100}
# output_dim 2 (either 0 or 1)
class BankNN(nn.Module):
    def __init__(self, num_features, hidden_layer_width, output_dim):
        super(BankNN, self).__init__()
        self.linear1 = nn.Linear(num_features, hidden_layer_width)
        self.linear2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        # Uncomment to add more layers
        self.linear3 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear4 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear5 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear6 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear7 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear8 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear9 = nn.Linear(hidden_layer_width, output_dim)

    # Change torch.tanh to torch.relu if using that activation method
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        x = torch.tanh(self.linear5(x))
        x = torch.tanh(self.linear6(x))
        x = torch.tanh(self.linear7(x))
        x = torch.tanh(self.linear8(x))
        x = torch.tanh(self.linear9(x))
        return torch.sigmoid(x)


class BankDataset(Dataset):
    def __init__(self, x_train, y_train):
        super(BankDataset, self).__init__()
        self.x = torch.from_numpy(x_train.astype(np.float32))
        self.y = torch.from_numpy(y_train.astype(np.float32))
        self.length = self.x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train(model, dataloader, device, optimizer):
    model.train()
    avg_loss = 0
    for _, (input, label) in enumerate(dataloader):
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device)
        output = model(input).squeeze(0)
        loss = nn.BCELoss()
        # print(output.squeeze(0).shape)
        # print(label.shape)
        loss = loss(output, label)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
    return avg_loss / len(dataloader)


def main():
    cols = ["variance", "skewness", "curtosis", "entropy", "label"]
    train_df = pd.read_csv("bank-note/train.csv", names=cols)
    X_train = train_df.drop(columns=["label"])
    X_train = X_train.to_numpy()
    Y_train = train_df["label"].to_numpy()
    dataset = BankDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, shuffle=False)

    train_df = pd.read_csv("bank-note/test.csv", names=cols)
    X_test = train_df.drop(columns=["label"])
    X_test = X_test.to_numpy()
    Y_test = train_df["label"].to_numpy()
    test_dataset = BankDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Don't change these
    num_features = 4
    output_dim = 1
    # Change these (width = {5, 10, 25, 50, 100}, num_layers = {3, 5, 9})
    hidden_layer_width = 25
    # Layers must be added manually, remember to update this value
    num_layers = 3
    model = BankNN(
        num_features=num_features,
        hidden_layer_width=hidden_layer_width,
        output_dim=output_dim,
    ).to(device)
    # Use xavier initialization for tanh
    nn.init.xavier_uniform_(model.linear1.weight)
    nn.init.xavier_uniform_(model.linear2.weight)
    # Make sure to comment/uncomment when adding/removing layers
    # nn.init.xavier_uniform_(model.linear3.weight)
    # nn.init.xavier_uniform_(model.linear4.weight)
    # nn.init.xavier_uniform_(model.linear5.weight)
    # nn.init.xavier_uniform_(model.linear6.weight)
    # nn.init.xavier_uniform_(model.linear7.weight)
    # nn.init.xavier_uniform_(model.linear8.weight)
    # Always last layer
    nn.init.xavier_uniform_(model.linear9.weight)

    # Use he initialization for RELU
    # nn.init.kaiming_uniform_(model.linear1.weight, nonlinearity="relu")
    # nn.init.kaiming_uniform_(model.linear2.weight, nonlinearity="relu")
    # Make sure to comment/uncomment when adding/removing layers
    # nn.init.kaiming_uniform_(model.linear3.weight, nonlinearity="relu")
    # nn.init.kaiming_uniform_(model.linear4.weight, nonlinearity="relu")
    # nn.init.kaiming_uniform_(model.linear5.weight, nonlinearity="relu")
    # nn.init.kaiming_uniform_(model.linear6.weight, nonlinearity="relu")
    # nn.init.kaiming_uniform_(model.linear7.weight, nonlinearity="relu")
    # nn.init.kaiming_uniform_(model.linear8.weight, nonlinearity="relu")
    # Always last layer
    # nn.init.kaiming_uniform_(model.linear9.weight, nonlinearity="relu")

    optimizer = torch.optim.Adam(model.parameters())
    loss = train(model, dataloader, device, optimizer)
    train_error = get_error(model, dataloader)
    test_error = get_error(model, test_dataloader)
    print(
        f"tanh & {num_layers} & {hidden_layer_width} & {train_error:.5f} & {test_error:.5f} \\\\\hline"
    )


def get_error(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    incorrect = 0
    for idx, (input, label) in enumerate(dataloader):
        # Need to unsqueeze for correct tensor shape
        input, label = input.to(device), label.to(device)
        output = torch.round(model(input))
        if output != label:
            incorrect += 1

    return incorrect / len(dataloader)


if __name__ == "__main__":
    main()
