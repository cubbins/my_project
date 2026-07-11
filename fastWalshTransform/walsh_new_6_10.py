import torch
import torch.nn as nn
import torch.nn.functional as F

def fwht(x):
    """
    In-place Fast Walsh–Hadamard Transform on the last dimension.
    x: (..., d) where d is a power of 2
    """
    orig_shape = x.shape
    d = x.size(-1)
    assert (d & (d - 1)) == 0, "Last dimension must be power of 2 for FWHT."

    h = 1
    while h < d:
        # x.view(..., d//(2*h), 2, h) pattern
        x = x.view(*orig_shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*orig_shape)
        h *= 2

    # Optional normalization (orthonormal)
    x = x / (d ** 0.5)
    return x

class WalshAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: (B, T, D) -> (B, H, T, d_head)
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # x: (B, H, T, d_head) -> (B, T, D)
        B, H, T, d_head = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * d_head)

    def forward(self, x, mask=None):
        """
        x: (B, T, D)
        mask: (B, 1, 1, T) or (B, 1, T, T) boolean, optional
        """
        B, T, D = x.shape

        Q = self._split_heads(self.q_proj(x))  # (B, H, T, d_head)
        K = self._split_heads(self.k_proj(x))  # (B, H, T, d_head)
        V = self._split_heads(self.v_proj(x))  # (B, H, T, d_head)

        # Apply Walsh–Hadamard on feature dimension (d_head)
        Qw = fwht(Q)
        Kw = fwht(K)

        # Compute similarity via elementwise product + sum over d_head
        # logits: (B, H, T, T)
        # We broadcast: Qw (B,H,T,1,d) * Kw (B,H,1,T,d)
        Qw_exp = Qw.unsqueeze(3)  # (B,H,T,1,d)
        Kw_exp = Kw.unsqueeze(2)  # (B,H,1,T,d)
        logits = (Qw_exp * Kw_exp).sum(-1) / (self.d_head ** 0.5)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to V (you can also Walsh-transform V if you want)
        out = torch.matmul(attn, V)  # (B, H, T, d_head)

        out = self._merge_heads(out)  # (B, T, D)
        out = self.o_proj(out)
        return out, attn

class WalshTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = WalshAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Attention + residual
        attn_out, _ = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out

        # FFN + residual
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        return x

