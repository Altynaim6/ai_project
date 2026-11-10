"""Standalone two-layer GCN classifier over a fixed HAI tag graph."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class HAIStandaloneGCN(nn.Module):
    def __init__(self, edge_index, node_static, hidden=32, embedding_dim=8, dropout=.1):
        super().__init__()
        if node_static.ndim != 2 or node_static.shape[1] != 8:
            raise ValueError("Expected process/role static features [nodes,8]")
        self.num_nodes = node_static.shape[0]
        if edge_index.shape[0] != 2 or edge_index.min() < 0 or edge_index.max() >= self.num_nodes:
            raise ValueError("Invalid graph indices")
        self.register_buffer("edge_index", edge_index.clone())
        self.register_buffer("node_static", node_static.clone())
        self.register_buffer("node_ids", torch.arange(self.num_nodes))
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        # Self-loops already exist exactly once in exported graph; never add them again.
        self.conv1 = GCNConv(6+8+embedding_dim, hidden, add_self_loops=False, cached=False)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=False, cached=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def encode_nodes(self, dynamic):
        if dynamic.ndim != 3 or dynamic.shape[1:] != (self.num_nodes, 6):
            raise ValueError(f"Expected [batch,{self.num_nodes},6]")
        batch = len(dynamic)
        static = self.node_static.unsqueeze(0).expand(batch, -1, -1)
        embedding = self.embedding(self.node_ids).unsqueeze(0).expand(batch, -1, -1)
        values = torch.cat([dynamic, static, embedding], dim=-1).reshape(batch*self.num_nodes, -1)
        offsets = torch.arange(batch, device=dynamic.device).reshape(-1,1,1)*self.num_nodes
        edges = (self.edge_index.unsqueeze(0)+offsets).permute(1,0,2).reshape(2,-1)
        values = self.dropout(torch.relu(self.conv1(values, edges)))
        values = torch.relu(self.conv2(values, edges))
        return values.reshape(batch, self.num_nodes, -1)

    def forward(self, dynamic):
        values = self.encode_nodes(dynamic)
        pooled = torch.cat([values.mean(dim=1), values.amax(dim=1)], dim=1)
        return self.classifier(pooled)
