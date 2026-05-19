import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import OptPairTensor, Adj, Size, NoneType, OptTensor

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_scatter import scatter


# ─────────────────────────────────────────────
# GNN Stack Module
# ─────────────────────────────────────────────

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers - 1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.post_mp(x)

        if self.emb:
            return x
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


# ─────────────────────────────────────────────
# GraphSAGE Layer
# ─────────────────────────────────────────────

class GraphSage(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = Linear(in_channels, out_channels, bias=bias)
        self.lin_r = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        # 1. Message passing (mean aggregation of neighbors)
        out = self.propagate(edge_index, x=(x, x), size=size)

        # 2. Skip connection: h_v = W_l * h_v + W_r * AGG(neighbors)
        out = self.lin_l(x) + self.lin_r(out)

        # 3. L-2 normalization
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return scatter(
            inputs,
            index,
            dim=self.node_dim,
            dim_size=dim_size,
            reduce='mean',
        )


# ─────────────────────────────────────────────
# GAT Layer
# ─────────────────────────────────────────────

class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Shared linear transformation applied before message passing
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_r = self.lin_l

        # Attention parameters (H, C)
        self.att_l = Parameter(torch.Tensor(heads, out_channels))
        self.att_r = Parameter(torch.Tensor(heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        H, C = self.heads, self.out_channels

        # 1. Linear projection + reshape → (N, H, C)
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        # 2. Per-node attention scores (N, H)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        # 3. Message passing
        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            alpha=(alpha_l, alpha_r),
            size=size,
        )  # (N, H, C)

        # 4. Flatten heads → (N, H*C)
        out = out.view(-1, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        # 1. e_ij = LeakyReLU(alpha_i + alpha_j)
        alpha = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)

        # 2. Softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)

        # 3. Dropout on attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4. Weighted message → (E, H, C)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None):
        return scatter(
            inputs,
            index,
            dim=self.node_dim,
            dim_size=dim_size,
            reduce='sum',
        )


# ─────────────────────────────────────────────
# Optimizer Builder
# ─────────────────────────────────────────────

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


# ─────────────────────────────────────────────
# Training & Testing
# ─────────────────────────────────────────────

def train(dataset, args):
    print("Node task. test set size:", np.sum(dataset[0]['train_mask'].numpy()))
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args)
    scheduler, opt = build_optimizer(args, model.parameters())

    losses = []
    test_accs = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
        else:
            test_accs.append(test_accs[-1])

    return test_accs, losses


def test(loader, model, is_validation=True):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        mask = data.val_mask if is_validation else data.test_mask
        correct += pred[mask].eq(data.y[mask]).sum().item()

    total = sum(
        torch.sum(data.val_mask if is_validation else data.test_mask).item()
        for data in loader.dataset
    )
    return correct / total


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

class objectview:
    def __init__(self, d):
        self.__dict__ = d


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    for args in [
        {
            'model_type': 'GraphSage',
            'dataset': 'cora',
            'num_layers': 2,
            'heads': 1,
            'batch_size': 32,
            'hidden_dim': 32,
            'dropout': 0.6,
            'epochs': 500,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'weight_decay': 5e-4,
            'lr': 0.01,
        },
    ]:
        args = objectview(args)
        for model_name in ['GraphSage', 'GAT']:
            args.model_type = model_name
            args.heads = 2 if model_name == 'GAT' else 1

            if args.dataset == 'cora':
                dataset = Planetoid(root='/tmp/cora', name='Cora')
            else:
                raise NotImplementedError("Unknown dataset")

            test_accs, losses = train(dataset, args)

            print(f"[{model_name}] Maximum accuracy: {max(test_accs):.4f}")
            print(f"[{model_name}] Minimum loss:     {min(losses):.4f}")

            plt.plot(losses, label=f"training loss - {model_name}")
            plt.plot(test_accs, label=f"test accuracy - {model_name}")

        plt.title(dataset.name)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
