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

#data = dataset[0]

# Print information about the dataset
#print(f'Number of graphs: {len(dataset)}')
#print(f'Number of nodes: {data.x.shape[0]}')
#print(f'Number of features: {dataset.num_features}')
#print(f'Number of classes: {dataset.num_classes}')
#print(f'Has isolated nodes: {data.has_isolated_nodes()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

# encoder class
class EncoderLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N, node_dim=0, use_dropout=True):
        # N is data.x[0]
        super().__init__(aggr='add', node_dim=node_dim) #  "Add" aggregation.
        self.mlp = nn.Linear(d_model, d_model, bias=True)
        # Weight matrices W_Q, W_K, W_V
        # linear layers to transform the feature vectors into query, key and value representations
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_k_sqrt = math.sqrt(d_k)
        # self.embed = nn.Linear(in_channels, d_model, bias=False)
      #  self.positional_encoding = PositionalEncoding(N,d_model)

        self.W_q = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * num_heads, bias=False)
        self.W_o = nn.Linear(num_heads * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_mlp = nn.LayerNorm(d_model)
        # initialise the parent 'messagepassing' class, specifying the aggregation method as 'add'
        self.use_dropout = use_dropout
        self.dropout = 0.6

    def split_heads(self, x):
        # print(x.size())
        batch_size, M = x.size()
        d = M / self.num_heads
        # print('d: {}'.format(d))
        d = int(d)
        # aft transpose, dimensions bs * N * sl * d_model
        return x.view(batch_size, self.num_heads, d).transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1)

    def forward(self, x, edge_index):
        # edge_index has shape [2, E]
        # x has shape [N, in_channels]
        # print('in encoder, x: {}'.format(x.shape))
        # x = self.embed(x)
       # x = self.positional_encoding(x)

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        # use attention mechanism for message passing. compute attention coefficients using Q, K and agg info using V
        # print('encoder, x: {}'.format(x.shape))
        # print('Q: {}'.format(Q.shape))
        # print('K: {}'.format(K.shape))
        # print('V: {}'.format(V.shape))

        h = self.propagate(edge_index, x=x, Q=Q, K=K, V=V)
        # pass aggregated msg through MLP layer, then apply layernorm
        h = h.view(h.size(0),-1) # ADDED BY XX, flatten for MLP
        h = h + self.mlp(h)
        output = self.layer_norm_mlp(h)
        return output

    def message(self, x_i, x_j, Q_i, K_j, V_j):
        # define how msgs are computed betweenin neighbouring nodes
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        # Q_i has dimensions [batch_size, num_heads, d_k]
        # K_j.transpose(1,2) has dimensions [batch_size, d_k, num_heads]
        alpha_ij = torch.matmul(Q_i, K_j.transpose(-2,-1))
        att_ij = torch.softmax(alpha_ij / self.d_k_sqrt, dim=-1)
        return torch.matmul(att_ij, V_j)

    def update(self, message, x):
        # message is already aggregated over the neighbors (message = sum_j m_ij)
        # node_attr is position encoding
        # print('message: {}'.format(message.shape))
        # print('x: {}'.format(x.shape))
        #h = x + self.W_o(self.combine_heads(message))
        attn_out = self.combine_heads(message)
        #attn_out = F.relu(attn_out)
        if self.use_dropout:
          attn_out = F.dropout(attn_out, self.dropout, training = self.training)

        h = x + self.W_o(attn_out)
        return self.layer_norm(h) #apply layernorm to tensor h, but normalise the tensor along the last dimension

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N, num_layers):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(in_channels, d_model, bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers-1):
          if i == 0:
            self.encoder_layers.append(EncoderLayer(in_channels, out_channels, d_k, d_model, d_v, num_heads, N))
          else:
            self.encoder_layers.append(EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N))


    def forward(self, x, edge_index):

        x = self.embed(x)
        # Encode
        for layer in self.encoder_layers:
            x = layer(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N, num_layers):
        super(GAT, self).__init__()

        # Encoder
        self.encoder = Encoder(in_channels, out_channels, d_k, d_model, d_v, num_heads, N, num_layers)

        # Decoder
        # self.decoder = Decoder(d_model, out_channels, d_k, d_model, d_v, num_heads, N)

        # Linear Layer
        self.linear = nn.Linear(d_model, out_channels)

    def forward(self, edge_index, x):
        # Encode
        encoder_output = self.encoder(x, edge_index)
        # print(edge_index.shape)
        # print('encoder output: ', encoder_output.shape)

        # Decode
        # decoder_output = self.decoder(edge_index,encoder_output)

        # Linear Layer
        output = self.linear(encoder_output)

        output = F.log_softmax(output, dim=-1)

        return output

in_channels = dataset.num_features
# print(in_channels)
out_channels = dataset.num_classes
# print(out_channels)
num_heads = 32
d_k = 16
d_model = 256 #default values
d_v = 16
N = data.x.shape[0]
num_layers = 8
num_epochs = 500
weight_decay = 1e-5
learning_rate = 0.0001

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--d_model', type=int, help='d_model value', required=True)
    #parser.add_argument('--num_layers', type=int, help='num of layers value', required=True)
    #parser.add_argument('--weight_decay', type=float, help='weight decay', required=True)
    #parser.add_argument('--learning_rate', type=float, help='learning rate', required=True)

   # args = parser.parse_args()
    
    #d_model = args.d_model
    #num_layers = args.num_layers
    #weight_decay = args.weight_decay
   # learning_rate = args.learning_rate
    d_v = d_model // num_heads
    d_k = d_v

    with open('results_pubmed.txt', 'a') as results_file:
        results_file.write(f"d_model: {d_model}\n")
        
    # Initialize lists to store accuracy values
    final_train_accuracies = []
    final_val_accuracies = []
    final_test_accuracies = []

    for i in range(1):
        seed = random.randint(1, 10000)
        with open('results_pubmed.txt', 'a') as results_file:
            results_file.write(f"run number {i+1}\n")
            results_file.write(f"seed: {seed}\n")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print("d_model is",d_model)
        print("d_k and d_v are", d_k, d_v)

        model = GAT(dataset.num_features, dataset.num_classes, d_k, d_model, d_v, num_heads,N, num_layers).to('cuda')
        data = dataset[0].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        arguments = {"learning_rate": learning_rate, "architecture": "gat", "dataset": name_data, "epochs":num_epochs}
        wandb_id = None
        run_name = "{}_{}weight_{}layers_{}dmodel_{}numheads_{}learningrate".format(name_data,weight_decay,num_layers,d_model,num_heads,learning_rate) 
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        wandb.init(project="GAT", name=run_name, config=arguments, entity="mli-kit", id=wandb_id, resume="allow")

        # New training loop with validation and test
        best_val_loss = float('inf')
        best_model = None
        train_losses = []
        val_losses = []
        val_accuracies = []
        train_accuracies = []

        def accuracy(pred_y, y):
            """Calculate accuracy."""
            return ((pred_y == y).sum() / len(y)).item()

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.edge_index, data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc*100:.2f}')
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            wandb.log({"epoch": epoch, "loss": loss, "train accuracy": acc})

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_output = model(data.edge_index, data.x)
                val_loss = criterion(val_output[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                val_losses.append(val_loss.item())
                print(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}')
                wandb.log({"epoch": epoch, "val loss": val_loss, "val accuracy": val_acc})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_train_acc = acc
                best_model = model.state_dict()

        # Load the best model and evaluate on the test set
        model.load_state_dict(best_model)
        model.eval()

        test_output = model(data.edge_index, data.x)
        test_loss = criterion(test_output[data.test_mask], data.y[data.test_mask])
        print(f"Test Loss: {test_loss:.4f}")
        test_acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
        print(f"Test Accuracy: {test_acc:.4f}")

        final_test_accuracies.append(test_acc)
        final_train_accuracies.append(best_train_acc)
        final_val_accuracies.append(best_val_acc)

        with open('results_pubmed.txt', 'a') as results_file:
            results_file.write(f"Test Loss: {test_loss:.4f} \n")
            results_file.write(f"Test Accuracy: {test_acc:.4f} \n")

    # Calculate means and standard deviations
    train_acc_mean = np.mean(final_train_accuracies)
    train_acc_std = np.std(final_train_accuracies)
    val_acc_mean = np.mean(final_val_accuracies)
    val_acc_std = np.std(final_val_accuracies)
    test_acc_mean = np.mean(final_test_accuracies)
    test_acc_std = np.std(final_test_accuracies)

    print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))

    with open('results_pubmed.txt', 'a') as results_file:
        results_file.write(f"d_model: {d_model}, num_heads: {num_heads}, seed: {seed}\n")

        results_file.write("means and standard deviations of the 3 runs \n")
        results_file.write(f"Train Accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}\n")
        results_file.write(f"Validation Accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}\n")
        results_file.write(f"Test Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}\n")
        results_file.write("\n")


if __name__ == "__main__":
    main()


# Plot training and validation loss
#plt.plot(train_losses, label='Train Loss')
#plt.plot(val_losses, label='Validation Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#matplotlib.pyplot.show()
#plt.show(block=False)
#plt.pause(1)
#input()
#plt.close()
#plt.savefig('plot.png')
#plt.close()


# dont use this
class PositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model, n= 10000):
        super(PositionalEncoding, self).__init__()

    # generate an empty matrix for the positional encodings (pe)
        pe = torch.zeros(max_length*d_model).reshape(max_length, d_model)
    # for each position
        for k in torch.arange(max_length):
      # for each dimension
          for i in torch.arange(d_model//2):
        # calculate the internal value for sin and cos
            theta = k / (n ** ((2*i)/d_model))
        # even dims: sin
            pe[k, 2*i] = math.sin(theta)
        # odd dims: cos
            pe[k, 2*i+1] = math.cos(theta)
        self.register_buffer('pe',pe)

    def forward(self,x):
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]

class MultiHeadAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N, node_dim=0):
        super().__init__(aggr='add', node_dim=node_dim) #  "Add" aggregation.

        self.d_model = d_model
        self.d_v = d_v
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_k_sqrt = math.sqrt(d_k)
        self.embed = nn.Linear(in_channels, d_model, bias=False)
        #self.positional_encoding = PositionalEncoding(N,d_model)

        self.W_q = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * num_heads, bias=False)
        self.W_o = nn.Linear(num_heads * d_v, d_model)

        self.dropout = 0.6

    def split_heads(self, x):
        # print(x.size())
        batch_size, M = x.size()
        d = M / self.num_heads
        # print('d: {}'.format(d))
        d = int(d)
        # aft transpose, dimensions bs * N * sl * d_model

        return x.view(batch_size, self.num_heads, d).transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1)

    def forward(self, edge_index, x):
        # edge_index has shape [2, E]
        # x has shape [N, in_channels]
        # print('decoder layer 1, x: {}'.format(x.shape))
        x = self.embed(x)
        # print('before pe, x: {}'.format(x.shape))

        #x = self.positional_encoding(x)
        # print('after decoder pe, x: {}'.format(x.shape))
        #add positional embedding

        # x is basically the embeddings.
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        # use attention mechanism for message passing. compute attention coefficients using Q, K and agg info using V
        # print('x: {}'.format(x.shape))
        # print('Q: {}'.format(Q.shape))
        # print('K: {}'.format(K.shape))
        # print('V: {}'.format(V.shape))
        h = self.propagate(edge_index, x=x, Q=Q, K=K, V=V)
        # pass aggregated msg through MLP layer, then apply layernorm
        h = h.view(h.size(0),-1)

        return h

    def message(self, x_i, x_j, Q_i, K_j, V_j):
        # define how msgs are computed between neighbouring nodes
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        # Q_i has dimensions [batch_size, num_heads, d_k]
        # K_j.transpose(1,2) has dimensions [batch_size, d_k, num_heads]
        alpha_ij = torch.matmul(Q_i, K_j.transpose(-2,-1))
        att_ij = torch.softmax(alpha_ij / self.d_k_sqrt, dim=-1)
        return torch.matmul(att_ij, V_j)

    def update(self, message, x):
        # message is already aggregated over the neighbors (message = sum_j m_ij)
        # node_attr is position encoding
        # print('message: {}'.format(message.shape))
        # print('x: {}'.format(x.shape))
        attn_out = self.combine_heads(message)
        #attn_out = F.relu(attn_out)

        # add a dropout (v3)
        attn_out = F.dropout(attn_out, self.dropout, training = self.training)

        h = x + self.W_o(attn_out)
        return h

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(in_channels, d_model, bias=False)
        # Encoder
        self.encoder_layer1 = EncoderLayer(in_channels, out_channels, d_k, d_model, d_v, num_heads, N)
        self.encoder_layer2 = EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.encoder_layer3 = EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.encoder_layer4 = EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.encoder_layer5 = EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.encoder_layer6 = EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.encoder_layer7 = EncoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)

    def forward(self, x, edge_index):

        x = self.embed(x)
        # Encode
        encoder_layer1_output = self.encoder_layer1(x, edge_index)
        # print(encoder_layer1_output.shape)
        # print('encoder output: ', encoder_output.shape)
        encoder_layer2_output = self.encoder_layer2(encoder_layer1_output, edge_index)
        encoder_layer3_output = self.encoder_layer3(encoder_layer2_output, edge_index)
        encoder_layer4_output = self.encoder_layer4(encoder_layer3_output, edge_index)
        encoder_layer5_output = self.encoder_layer5(encoder_layer4_output, edge_index)
        encoder_layer6_output = self.encoder_layer6(encoder_layer5_output, edge_index)
        encoder_layer7_output = self.encoder_layer7(encoder_layer6_output, edge_index)

        return encoder_layer7_output

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N):
        super(Decoder, self).__init__()

        self.decoder_layer1 = DecoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.decoder_layer2 = DecoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.decoder_layer3 = DecoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.decoder_layer4 = DecoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)
        self.decoder_layer5 = DecoderLayer(d_model, out_channels, d_k, d_model, d_v, num_heads, N)

    def forward(self, edge_index, x):

        decoder_layer1_output = self.decoder_layer1(edge_index, x)
        # print(edge_index.shape)
        # print('encoder output: ', encoder_output.shape)
        decoder_layer2_output = self.decoder_layer2(edge_index,decoder_layer1_output)

        decoder_layer3_output = self.decoder_layer3(edge_index,decoder_layer2_output)
        decoder_layer4_output = self.decoder_layer4(edge_index,decoder_layer3_output)
        decoder_layer5_output = self.decoder_layer5(edge_index,decoder_layer4_output)
        return decoder_layer5_output

# decoder class
# 2 multihead attention layers, 1 mlp feedforward layer, 2 layer norm layers

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, d_k, d_model, d_v, num_heads, N):
        super(DecoderLayer, self).__init__()

        # First Multi-Head Attention Layer
        self.multihead_attention1 = MultiHeadAttention(in_channels, out_channels, d_k, d_model, d_v, num_heads, N, node_dim=0)
        self.layer_norm1 = nn.LayerNorm(d_model)

        # Second Multi-Head Attention Layer
        self.multihead_attention2 = MultiHeadAttention(in_channels, out_channels, d_k, d_model, d_v, num_heads, N, node_dim=0)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Feed Forward Layer
        self.mlp = nn.Linear(d_model, d_model, bias=True)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, edge_index, x):

        # First Multi-Head Attention Layer
        # print('edge index:', edge_index.shape)
        attention_output1 = self.multihead_attention1(edge_index, x)
        # print(attention_output1.shape)
        x = x + attention_output1
        # print(x.shape)
        x = self.layer_norm1(x)

        # Second Multi-Head Attention Layer
        attention_output2 = self.multihead_attention2(edge_index, x)
        x = x + attention_output2
        x = self.layer_norm2(x)

        # Feed Forward Layer
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.layer_norm3(x)

        return x
