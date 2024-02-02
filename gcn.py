import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import math
import tqdm
#import matplotlib.pyplot as plt
import wandb
import argparse
import random

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

#name_data = 'Cora'
#name_data = 'PubMed'
name_data = 'CiteSeer'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(data.num_features, 1024)
        self.conv2 = GCNConv(1024, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



# useful function for computing accuracy
def compute_accuracy(pred_y, y):
    return (pred_y == y).sum()
def accuracy(pred_y, y):
  return ((pred_y == y).sum() / len(y)).item()
        
    # Initialize lists to store accuracy values
final_train_accuracies = []
final_val_accuracies = []
final_test_accuracies = []

for i in range(3):
    seed = random.randint(1, 10000)
    with open('results_gcn.txt', 'a') as results_file:
        results_file.write(f"run number {i+1}\n")
        results_file.write(f"seed: {seed}\n")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    best_val_loss = float('inf')
    best_model = None    
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    losses = []
    accuracies = []
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        correct = compute_accuracy(out.argmax(dim=1)[data.train_mask], data.y[data.train_mask])
        acc = int(correct) / int(data.train_mask.sum())
        train_losses.append(loss.item())
        accuracies.append(acc)
        loss.backward()
        optimizer.step()
        print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch+1, loss.item(), acc))

# evaluate the model on test set
        model.eval()

        with torch.no_grad():
            val_output = model(data)
            val_loss = F.nll_loss(val_output[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(val_output[data.val_mask].argmax(dim=1), data.y[data.val_mask])
            val_losses.append(val_loss.item())
            print(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_acc = acc
            best_model = model.state_dict()
    
    model.load_state_dict(best_model)
    model.eval()

    pred = model(data)
    test_loss = F.nll_loss(pred[data.test_mask], data.y[data.test_mask])
    acc = accuracy(pred[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    print(f'test Accuracy: {acc:.4f}')

    final_test_accuracies.append(acc)
    final_train_accuracies.append(best_train_acc)
    final_val_accuracies.append(best_val_acc)

    with open('results_gcn.txt', 'a') as results_file:
        results_file.write(f"Test Loss: {test_loss:.4f} \n")
        results_file.write(f"Test Accuracy: {acc:.4f} \n")

train_acc_mean = np.mean(final_train_accuracies)
train_acc_std = np.std(final_train_accuracies)
val_acc_mean = np.mean(final_val_accuracies)
val_acc_std = np.std(final_val_accuracies)
test_acc_mean = np.mean(final_test_accuracies)
test_acc_std = np.std(final_test_accuracies)

print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))


with open('results_gcn.txt', 'a') as results_file:

    results_file.write("means and standard deviations of the 3 runs \n")
    results_file.write(f"Train Accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}\n")
    results_file.write(f"Validation Accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}\n")
    results_file.write(f"Test Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}\n")
    results_file.write("\n")