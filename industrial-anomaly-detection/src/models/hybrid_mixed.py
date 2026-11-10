"""Fixed-graph GCN per observation followed by a causal temporal Transformer."""
import torch
from torch import nn
from src.models.transformer_mixed import MixedForecastTransformer


class HybridMixedForecast(MixedForecastTransformer):
    def __init__(self, edge_index, node_static, graph_to_input, node_order,
                 graph_sha256, graph_hidden=16, embedding_dim=8,
                 num_numeric=55, num_binary=4, sequence_length=30,
                 d_model=64, nhead=4, num_layers=2, dropout=.1):
        super().__init__(num_numeric, num_binary, sequence_length, d_model,
                         nhead, num_layers, dropout)
        n = self.num_features
        edges = torch.as_tensor(edge_index, dtype=torch.long)
        static = torch.as_tensor(node_static, dtype=torch.float32)
        mapping = torch.as_tensor(graph_to_input, dtype=torch.long)
        if (graph_hidden < 1 or embedding_dim < 1 or static.shape != (n, 8)
                or not torch.isfinite(static).all()
                or mapping.shape != (n,) or sorted(mapping.tolist()) != list(range(n))
                or len(node_order) != n or len(set(node_order)) != n):
            raise ValueError('Invalid graph nodes, input permutation or dimensions')
        if edges.ndim != 2 or edges.shape[0] != 2 or not edges.numel() or edges.min() < 0 or edges.max() >= n:
            raise ValueError('Invalid graph edges')
        pairs = set(map(tuple, edges.T.tolist()))
        if (len(pairs) != edges.shape[1] or any((b,a) not in pairs for a,b in pairs)
                or sum(a == b for a,b in pairs) != n):
            raise ValueError('Expected symmetric unique edges with one self-loop per node')
        if not isinstance(graph_sha256, str) or len(graph_sha256) != 64:
            raise ValueError('Graph provenance must be a SHA256 string')
        # GCN normalization D^-1/2 A D^-1/2. For this fixed 59-node graph,
        # dense batched matmul avoids constructing B*T copies of edge indices.
        adjacency = torch.zeros(n, n)
        adjacency[edges[1], edges[0]] = 1.
        inverse = adjacency.sum(1).rsqrt()
        normalized = inverse[:, None]*adjacency*inverse[None, :]
        self.register_buffer('normalized_adjacency', normalized)
        self.register_buffer('node_static', static.clone())
        self.register_buffer('graph_to_input', mapping.clone())
        self.node_embedding = nn.Embedding(n, embedding_dim)
        self.spatial1 = nn.Linear(1+8+embedding_dim, graph_hidden, bias=False)
        self.spatial2 = nn.Linear(graph_hidden, graph_hidden, bias=False)
        self.spatial_bias1 = nn.Parameter(torch.zeros(graph_hidden))
        self.spatial_bias2 = nn.Parameter(torch.zeros(graph_hidden))
        self.spatial_dropout = nn.Dropout(dropout)
        # Flattening keeps each device's identity in the temporal token.
        self.input_projection = nn.Linear(n*graph_hidden, d_model)
        self.model_config.update(edge_index=edges.tolist(), node_static=static.tolist(),
            graph_to_input=mapping.tolist(), node_order=list(node_order),
            graph_sha256=graph_sha256, graph_hidden=graph_hidden, embedding_dim=embedding_dim)

    def spatial_encode(self, history):
        if history.ndim != 3 or history.shape[-1] != self.num_features:
            raise ValueError(f'Expected [batch,time,{self.num_features}]')
        b, t, _ = history.shape
        values = history.index_select(-1, self.graph_to_input).unsqueeze(-1)
        static = self.node_static.expand(b, t, -1, -1)
        embedding = self.node_embedding.weight.expand(b, t, -1, -1)
        hidden = torch.cat([values, static, embedding], dim=-1)
        hidden = torch.relu(self.normalized_adjacency @ self.spatial1(hidden)+self.spatial_bias1)
        hidden = self.spatial_dropout(hidden)
        return torch.relu(self.normalized_adjacency @ self.spatial2(hidden)+self.spatial_bias2)

    def encode(self, history):
        if history.ndim != 3 or not 1 <= history.shape[1] <= self.sequence_length:
            raise ValueError('Invalid history length')
        length = history.shape[1]
        spatial = self.spatial_encode(history).flatten(2)
        hidden = self.input_projection(spatial)+self.positional_encoding[:, :length]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                    device=history.device), diagonal=1)
        return self.encoder(hidden, mask=mask)
