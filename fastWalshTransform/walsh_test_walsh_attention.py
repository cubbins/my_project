import torch
import torch.nn as nn

###############################################
# Fast Walsh–Hadamard Transform (FWHT)
###############################################

def fwht(x):
    """
    In-place Fast Walsh–Hadamard Transform on the last dimension.
    x: (..., d) where d is a power of 2
    """
    orig_shape = x.shape
    d = x.size(-1)
    assert (d & (d - 1)) == 0, "Last dimension must be power of 2."

    h = 1
    while h < d:
        x = x.view(*orig_shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*orig_shape)
        h *= 2

    return x / (d ** 0.5)


###############################################
# Walsh Attention Layer
###############################################

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
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * d)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        # Walsh–Hadamard mixing
        Qw = fwht(Q)
        Kw = fwht(K)

        # Similarity via elementwise product
        Qw_exp = Qw.unsqueeze(3)
        Kw_exp = Kw.unsqueeze(2)
        logits = (Qw_exp * Kw_exp).sum(-1) / (self.d_head ** 0.5)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = self._merge_heads(out)
        out = self.o_proj(out)

        return out, attn


###############################################
# Transformer Block using Walsh Attention
###############################################

class WalshTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = WalshAttention(d_model, n_heads, dropout)
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
        attn_out, _ = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


###############################################
# MAIN TEST
###############################################

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2      # batch size
    T = 8      # sequence length
    D = 32     # model dimension (must be divisible by n_heads)
    H = 4      # number of heads

    print("Creating dummy input...")
    x = torch.randn(B, T, D)

    print("Testing FWHT...")
    test_vec = torch.randn(1, 1, 8)  # 8 = power of 2
    print("FWHT input:", test_vec)
    print("FWHT output:", fwht(test_vec))

    print("\nTesting WalshAttention...")
    attn = WalshAttention(D, H)
    out, weights = attn(x)
    print("Output shape:", out.shape)
    print("Attention weights shape:", weights.shape)

    print("\nTesting WalshTransformerBlock...")
    block = WalshTransformerBlock(D, H)
    out2 = block(x)
    print("Block output shape:", out2.shape)

    print("\nAll tests completed successfully.")
