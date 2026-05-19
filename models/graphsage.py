import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class GraphSage(MessagePassing):
    """GraphSAGE layer with mean aggregation and skip connection.

    h_v = W_l * h_v + W_r * mean({h_u | u in N(v)})
    """

    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_l = Linear(in_channels, out_channels, bias=bias)  # central node
        self.lin_r = Linear(in_channels, out_channels, bias=bias)  # aggregated neighbors
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        agg = self.propagate(edge_index, x=(x, x), size=size)
        out = self.lin_l(x) + self.lin_r(agg)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
