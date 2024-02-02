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
import torch.nn.functional as F

#name_data = 'Cora'
name_data = 'PubMed'
#name_data = 'CiteSeer'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATConv(dim_in, dim_h, heads=heads)
    self.gat2 = GATConv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.0001,
                                      weight_decay=1e-5)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

final_train_accuracies = []
final_val_accuracies = []
final_test_accuracies = []

for i in range(3):
    seed = random.randint(1, 10000)    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Provide the dimensions when creating the GAT instance
    model = GAT(dataset.num_features, 512, dataset.num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    best_val_loss = float('inf')

    best_model = None    
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    losses = []
    accuracies = []
    epochs = 500

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        train_losses.append(loss.item())
        accuracies.append(acc)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            _, val_output = model(data.x, data.edge_index)
            val_loss = criterion(val_output[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(val_output[data.val_mask].argmax(dim=1), data.y[data.val_mask])
            val_losses.append(val_loss.item())
            print(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_acc = acc
            best_model = model.state_dict()      
        # Write metrics to file every 10 epochs
       # if epoch % 10 == 0:
       #     result_str = f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: ' \
        #                    f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | ' \
        #                    f'Val Acc: {val_acc*100:.2f}%'
        #    file.write(result_str + '\n')

    model.load_state_dict(best_model)
    model.eval()

    _, pred = model(data.x, data.edge_index)
    test_loss = criterion(pred[data.test_mask], data.y[data.test_mask])
    acc = accuracy(pred[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    print(f'test Accuracy: {acc:.4f}')

    final_test_accuracies.append(acc)
    final_train_accuracies.append(best_train_acc)
    final_val_accuracies.append(best_val_acc)
    test_result_str = f'Test Accuracy: {acc*100:.2f}%'
    with open('results_gat_pubmed.txt', 'w') as file:
        file.write(test_result_str + '\n')

train_acc_mean = np.mean(final_train_accuracies)
train_acc_std = np.std(final_train_accuracies)
val_acc_mean = np.mean(final_val_accuracies)
val_acc_std = np.std(final_val_accuracies)
test_acc_mean = np.mean(final_test_accuracies)
test_acc_std = np.std(final_test_accuracies)

print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))

with open('results_gat_pubmed.txt', 'w') as file:
    file.write(f'\nMean Train Accuracy: {train_acc_mean*100:.2f}% | Train Accuracy Std: {train_acc_std*100:.2f}%\n')
    file.write(f'Mean Validation Accuracy: {val_acc_mean*100:.2f}% | Validation Accuracy Std: {val_acc_std*100:.2f}%\n')
    file.write(f'Mean Test Accuracy: {test_acc_mean*100:.2f}% | Test Accuracy Std: {test_acc_std*100:.2f}%\n')