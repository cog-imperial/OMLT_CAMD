import math
import sys
import torch
from torch_geometric.data import Data, Dataset
import os.path as osp
import os


class MyOwnDataset(Dataset):
    def __init__(
        self, root, length, transform=None, pre_transform=None, pre_filter=None
    ):
        self.root = root
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        return self.length

    def get(self, idx):
        data = torch.load(osp.join(self.root, f"data_{idx}.pt"))
        return data


dataset = MyOwnDataset(root="data/garlic", length=262)

print(f"Number of node features: {dataset.num_node_features}")

import random

seed_gnn = int(sys.argv[1])

torch.manual_seed(seed_gnn)
dataset_shuffle = dataset.shuffle()

train_dataset = dataset_shuffle[:210]
test_dataset = dataset_shuffle[210:]

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# for step, data in enumerate(train_loader):
#    print(data)

from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import Sequential, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool

hidden_channels = 16
torch.manual_seed(seed_gnn)
model = Sequential(
    "x, edge_index, batch",
    [
        (
            SAGEConv(dataset.num_features, hidden_channels, aggr="sum"),
            "x, edge_index -> x",
        ),
        ReLU(inplace=True),
        (SAGEConv(hidden_channels, hidden_channels, aggr="sum"), "x, edge_index -> x"),
        ReLU(inplace=True),
        (global_mean_pool, "x, batch -> x"),
        Linear(hidden_channels, 2),
    ],
)
# print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    for data in train_loader:  # iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # perform a single forward pass.
        loss = criterion(out, data.y)  # compute the loss.
        loss.backward()  # derive gradients.
        optimizer.step()  # update parameters based on gradients.
        optimizer.zero_grad()  # clear gradients.


def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # use the class with highest probability.
        correct += int((pred == data.y).sum())  # check against ground-truth labels.
    return correct / len(loader.dataset)  # derive ratio of correct predictions.


for epoch in range(200):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    # print(f'Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    if epoch == 199:
        print(
            f"Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

# for parameter in model.parameters():
#    print(parameter)

dir = "model/garlic/"
if not os.path.exists(dir):
    os.makedirs(dir)
torch.save(model.state_dict(), os.path.join(dir, f"GNN_{seed_gnn}.pt"))
