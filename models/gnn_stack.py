import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphsage import GraphSage
from .gat import GAT


class GNNStack(torch.nn.Module):
    """Generic GNN stack that accepts GraphSage or GAT as the conv layer."""

    _CONV = {'GraphSage': GraphSage, 'GAT': GAT}

    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super().__init__()
        assert args.num_layers >= 1
        conv = self._CONV[args.model_type]

        self.convs = nn.ModuleList([conv(input_dim, hidden_dim)])
        for _ in range(args.num_layers - 1):
            self.convs.append(conv(args.heads * hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.emb = emb

    def forward(self, data):
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = F.dropout(F.elu(conv(x, edge_index)), p=self.dropout)

        x = self.post_mp(x)
        return x if self.emb else F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
