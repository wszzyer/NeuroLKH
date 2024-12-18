# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from net.multihead_attention import MultiheadAttention

class DoubleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0, 
        device: Optional[str] = None
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.module_list = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.GELU(),
        ])
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        self.module_list.append(nn.Linear(hidden_dim, output_dim, device=device))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.module_list:
            x = module(x)
        return x

class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int = 128,
        edge_embedding_dim: int = 64,
        ffn_embedding_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        device: Optional[str] = None
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout_module = nn.Dropout(dropout)

        # Initialize blocks
        self.self_attn = MultiheadAttention(node_embedding_dim, num_attention_heads, dropout=attention_dropout, device=device)
        self.node_ffn = DoubleLinear(node_embedding_dim, ffn_embedding_dim, dropout=activation_dropout, device=device)
        self.edge_fmap = nn.Linear(edge_embedding_dim, node_embedding_dim, device=device)
        self.edge_activitation = nn.GELU()
        self.edge_bmap = nn.Linear(node_embedding_dim, edge_embedding_dim, device=device)

        # layer norm associated with the self attention layer
        self.node_attn_norm = nn.LayerNorm((node_embedding_dim, ), device=device)
        self.edge_attn_norm = nn.LayerNorm((node_embedding_dim, ), device=device)

        # layer norm associated with the position wise feed-forward NN
        self.node_final_norm = nn.LayerNorm((node_embedding_dim, ), device=device)
        self.edge_final_norm = nn.LayerNorm((edge_embedding_dim, ), device=device)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        node_count, batch_size, hidden_size = x.shape
        # Attention part
        # x: T x B x C
        residual = x
        xt = x.transpose(1, 0)
        # attn_mask = torch.zeros((batch_size, node_count, node_count), dtype=torch.bool, device=x.device)
        batch_index = torch.arange(batch_size).reshape(-1, 1, 1)
        node_index = torch.arange(node_count).reshape(1, -1, 1)
        # attn_mask[batch_index, node_index, edge_index] = 1
        # attn_bias = torch.zeros_like(attn_mask, dtype=torch.float32)
        attn_bias = torch.zeros((batch_size, node_count, node_count), dtype=torch.float32, device=x.device)
        # The bias will be masked as well so do not worry
        attn_bias[batch_index, node_index, edge_index] = e.mean(dim=-1)
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=attn_bias,
            need_weights=True,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.node_attn_norm(x)

        # MLP part
        residual = x
        x = self.node_ffn(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.node_final_norm(x)
        
        # Edge part
        residual = e
        e = self.edge_fmap(e)
        weights = attn_weights.mean(dim=1).unsqueeze(-1)
        e = e + weights[batch_index, edge_index, node_index] * xt[batch_index.reshape(-1, 1), edge_index.flatten(1)].reshape(batch_size, node_count, -1, hidden_size)
        e = self.edge_attn_norm(e)
        e = self.edge_activitation(e)
        e = self.edge_bmap(e)
        e = residual + e
        e = self.edge_final_norm(e)

        return x, e

class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 12,
        node_embedding_dim: int = 128,
        edge_embedding_dim: int = 64,
        ffn_embedding_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        devices: Optional[List[str]] = None
    ) -> None:

        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.layerdrop = layerdrop
        self.devices = devices
        self.device_cap = (num_encoder_layers - 1) // len(devices) + 1 if devices else num_encoder_layers
        node_embedding_dim = node_embedding_dim

        # LayerDrop Removed.
        self.layers = nn.ModuleList(
            [
                GraphEncoderLayer(
                    node_embedding_dim=node_embedding_dim,
                    edge_embedding_dim=edge_embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    device=devices[index // self.device_cap]
                )
                for index in range(num_encoder_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: B x T x C

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Ugly codes for pipeline
        for index, layer in enumerate(self.layers):
            if self.devices and index % self.device_cap == 0:
                device = self.devices[index // self.device_cap]
                x = x.to(device)
                e = e.to(device)
                key_padding_mask = key_padding_mask.to(device)
                edge_index = edge_index.to(device)
            layer.to(device)
            x, e = layer(
                x,
                e,
                key_padding_mask=key_padding_mask,
                attn_mask=None, # We may use alpha-values to strengthen the attention, if only I have enough time
                edge_index=edge_index,
            )
            print(f"Layer {index}: {torch.cuda.memory_allocated(device) / 1e6:.1f}MiB")
        return x, e
