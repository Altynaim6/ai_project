"""Causal next-state Transformer with numeric and categorical forecast heads."""
import math

import torch
from torch import nn


class MixedForecastTransformer(nn.Module):
    def __init__(self, num_numeric=55, num_binary=4, sequence_length=30,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        if min(num_numeric, num_binary, sequence_length, d_model, nhead, num_layers) < 1:
            raise ValueError("Dimensions must be positive")
        if d_model % nhead or d_model % 2:
            raise ValueError("d_model must be even and divisible by nhead")
        self.model_config = dict(
            num_numeric=num_numeric, num_binary=num_binary,
            sequence_length=sequence_length, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout,
        )
        self.num_binary = num_binary
        self.num_features = num_numeric + num_binary
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(self.num_features, d_model)
        position = torch.arange(sequence_length).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))
        encoding = torch.zeros(sequence_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("positional_encoding", encoding.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        # TransformerEncoder clones initial layers; initialize each independently.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.numeric_head = nn.Linear(d_model, num_numeric)
        self.binary_head = nn.Linear(d_model, num_binary * 2)

    def encode(self, history):
        if history.ndim != 3 or history.shape[-1] != self.num_features:
            raise ValueError(f"Expected [batch, time, {self.num_features}]")
        length = history.shape[1]
        if not 1 <= length <= self.sequence_length:
            raise ValueError("Invalid history length")
        hidden = self.input_projection(history) + self.positional_encoding[:, :length]
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=history.device), diagonal=1
        )
        return self.encoder(hidden, mask=mask)

    def forward(self, history):
        hidden = self.encode(history)[:, -1]
        return (
            self.numeric_head(hidden),
            self.binary_head(hidden).reshape(-1, self.num_binary, 2),
        )
