# An Multi-head attention implementation adopted from pytorch.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import MultiheadAttention


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        device=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        assert self.qkv_same_dim, (
            "We have not check cross-attention yet!"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, device=device)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, device=device)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor],
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:

        tgt_len, batch_size, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert query.shape == (tgt_len, batch_size, embed_dim)
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == batch_size
                assert value is not None
                assert src_len, batch_size == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, batch_size * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, batch_size * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if attn_mask is not None:
            attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill_(~attn_mask, float("-inf"))
        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, src_len), \
                f"expecting key_padding_mask shape of {(batch_size, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask
        if attn_bias is not None:
            attn_bias = (attn_bias.view(batch_size, 1, src_len, src_len)
                         .expand(-1, self.num_heads, -1, -1)
                         .reshape(batch_size * self.num_heads, src_len, src_len))
            if attn_mask is not None:
                attn_mask = attn_mask + attn_bias # Cannot use += here
            else:
                attn_mask = attn_bias
        
        # We cannot use torch.nn.functional.scaled_dot_product_attention() as it applies a softmax automatically.
        if attn_mask is None:
            raw_qk = torch.bmm(q, k.transpose(1, 2))  
        else:
            raw_qk = torch.baddbmm(attn_mask, q, k.transpose(1, 2))
        assert raw_qk.shape == (batch_size * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(raw_qk, dim=-1)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert attn.shape == (batch_size * self.num_heads, tgt_len, self.head_dim)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            return attn, attn_weights
        else:
            return attn
