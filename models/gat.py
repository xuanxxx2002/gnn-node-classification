import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter


class GAT(MessagePassing):
    """Graph Attention Network layer with multi-head attention.

    alpha_ij = softmax_j( LeakyReLU( a_l^T W_l h_i + a_r^T W_r h_j ) )
    h_i'     = concat_k( sum_j alpha_ij^k * W_r^k h_j )
    """

    def __init__(self, in_channels, out_channels, heads=2,
                 negative_slope=0.2, dropout=0., **kwargs):
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_r = self.lin_l  # shared projection

        self.att_l = Parameter(torch.Tensor(heads, out_channels))
        self.att_r = Parameter(torch.Tensor(heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(x).view(-1, H, C)   # (N, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        alpha_l = (x_l * self.att_l).sum(dim=-1)  # (N, H)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        out = self.propagate(
            edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size
        )  # (N, H, C)
        return out.view(-1, H * C)  # (N, H*C)

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)  # (E, H, C)

    def aggregate(self, inputs, index, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
