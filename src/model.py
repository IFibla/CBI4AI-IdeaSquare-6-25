import torch
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weights = data.edge_attr

        edge_weight1 = edge_weights[:, 0]
        edge_weight2 = edge_weights[:, 1]

        x = self.conv1(x, edge_index, edge_weight1)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight2)
        return x
