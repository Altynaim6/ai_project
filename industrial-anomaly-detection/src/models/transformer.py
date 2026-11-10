import math

import torch
from torch import nn


class ForecastTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        sequence_length=30,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if d_model % 2 != 0:
            raise ValueError("Use an even d_model for positional encoding")

        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(num_features, d_model)

        positions = torch.arange(sequence_length).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(sequence_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)

        self.register_buffer(
            "positional_encoding",
            encoding.unsqueeze(0),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.forecast_head = nn.Linear(d_model, num_features)

    def forward(self, history):
        if history.ndim != 3:
            raise ValueError("Expected [batch, time, features]")

        length = history.shape[1]
        if not 1 <= length <= self.sequence_length:
            raise ValueError("Invalid history length")

        hidden = self.input_projection(history)
        hidden = hidden + self.positional_encoding[:, :length]

        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=history.device,
            ),
            diagonal=1,
        )

        hidden = self.encoder(hidden, mask=causal_mask)

        return self.forecast_head(hidden[:, -1])