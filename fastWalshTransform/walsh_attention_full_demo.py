import torch
import torch.nn as nn
import torch.nn.functional as F

# python walsh_attention_full_demo.py


# ============================================================
# 0. Walsh Matrix Utilities
# ============================================================

def walsh_matrix(n):
    """
    Explicitly generate the n x n Walsh–Hadamard matrix.
    n must be a power of 2.
    """
    if n == 1:
        return torch.tensor([[1.0]])
    H = walsh_matrix(n // 2)
    top = torch.cat([H, H], dim=1)
    bottom = torch.cat([H, -H], dim=1)
    return torch.cat([top, bottom], dim=0)

def print_walsh_matrix(n):
    print(f"\n==================== Walsh Matrix H_{n} ====================")
    H = walsh_matrix(n)
    print(H)


# ============================================================
# 1. Fast Walsh–Hadamard Transform (FWHT) with verbose printing
# ============================================================

def fwht(x, verbose=False):
    """
    Fast Walsh–Hadamard Transform over the last dimension.
    Uses reshape() instead of view() to support non-contiguous tensors.
    If verbose=True, prints intermediate butterfly stages.
    """
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])  # [N, D]
    N, D = x.shape
    h = 1
    stage = 0

    while h < D:
        if verbose:
            print(f"\n--- FWHT Stage {stage}, h={h} ---")
            print("Before reshape:\n", x)

        x = x.reshape(N, -1, 2 * h)
        a = x[:, :, :h]
        b = x[:, :, h:2*h]
        x = torch.cat([a + b, a - b], dim=-1)

        if verbose:
            print("After butterfly:\n", x)

        h *= 2
        stage += 1

    x = x.reshape(*orig_shape)
    return x / (orig_shape[-1] ** 0.5)


# ============================================================
# 2. WalshAttention Block
# ============================================================

class WalshAttention(nn.Module):
    def __init__(self, d_model, n_heads, verbose=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.verbose = verbose

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv=None):
        if kv is None:
            kv = x

        B, Tq, D = x.shape
        Tk = kv.size(1)

        q = self.W_q(x)
        k = self.W_k(kv)
        v = self.W_v(kv)

        if self.verbose:
            print("\n=== Raw Q ===\n", q)
            print("\n=== Raw K ===\n", k)

        q = q.view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)

        q = fwht(q, verbose=self.verbose)
        k = fwht(k, verbose=self.verbose)

        if self.verbose:
            print("\n=== Walsh-Transformed Q ===\n", q)
            print("\n=== Walsh-Transformed K ===\n", k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.W_o(out)
        return out


# ============================================================
# 3. Walsh-Transformer-XL
# ============================================================

class WalshTXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, mem_len, verbose=False):
        super().__init__()
        self.attn = WalshAttention(d_model, n_heads, verbose=verbose)
        self.mem_len = mem_len

    def forward(self, x, mem):
        if mem is None:
            cat = x
        else:
            cat = torch.cat([mem.detach(), x], dim=1)

        out = self.attn(x, kv=cat)
        new_mem = cat[:, -self.mem_len:].detach()
        return out, new_mem


# ============================================================
# 4. Walsh-Perceiver IO
# ============================================================

class WalshCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, verbose=False):
        super().__init__()
        self.attn = WalshAttention(d_model, n_heads, verbose=verbose)

    def forward(self, q, x):
        return self.attn(q, kv=x)

class WalshPerceiverIOStep(nn.Module):
    def __init__(self, d_model, n_heads, latent_len, verbose=False):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_len, d_model))
        self.in2lat = WalshCrossAttention(d_model, n_heads, verbose=verbose)
        self.lat2out = WalshCrossAttention(d_model, n_heads, verbose=verbose)

    def forward(self, x, q_out):
        B = x.size(0)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        z = self.in2lat(lat, x)
        y = self.lat2out(q_out, z)
        return y


# ============================================================
# 5. Walsh-Longformer (Sliding Window)
# ============================================================

def walsh_sliding_window_attention(q, k, v, w):
    T, D = q.size()
    q_w = fwht(q.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    k_w = fwht(k.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    out = torch.zeros_like(q)
    for i in range(T):
        left = max(0, i - w)
        right = min(T, i + w + 1)
        k_slice = k_w[left:right]
        v_slice = v[left:right]
        scores = (q_w[i:i+1] @ k_slice.T) / (D ** 0.5)
        attn = scores.softmax(dim=-1)
        out[i] = attn @ v_slice
    return out


# ============================================================
# 6. Walsh-FlashAttention-like
# ============================================================

def walsh_flash_like(q, k, v, block_size=4):
    T, D = q.size()

    q_w = fwht(q.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    k_w = fwht(k.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    o = torch.zeros_like(q)
    m = torch.full((T,), -1e9, device=q.device)
    l = torch.zeros(T, device=q.device)

    for start in range(0, T, block_size):
        end = min(T, start + block_size)
        k_blk = k_w[start:end]
        v_blk = v[start:end]

        scores = (q_w @ k_blk.T) / (D ** 0.5)

        m_new = torch.maximum(m, scores.max(dim=-1).values)
        l = torch.exp(m - m_new) * l + torch.exp(scores - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new

        weights = torch.exp(scores - m.unsqueeze(-1))
        o = o + weights @ v_blk

    o = o / l.unsqueeze(-1)
    return o


# ============================================================
# MAIN TEST DRIVER
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # --------------------------------------------------------
    # Print Walsh matrices
    # --------------------------------------------------------
    print_walsh_matrix(4)
    print_walsh_matrix(8)

    # --------------------------------------------------------
    # WalshAttention (verbose)
    # --------------------------------------------------------
    print("\n==================== WalshAttention (Verbose) ====================")
    B, T, D = 1, 4, 8
    walsh_attn = WalshAttention(d_model=D, n_heads=2, verbose=True)
    x = torch.randn(B, T, D)
    out = walsh_attn(x)
    print("WalshAttention output shape:", out.shape)

    # --------------------------------------------------------
    # Walsh-Transformer-XL
    # --------------------------------------------------------
    print("\n==================== WalshTXLBlock ====================")
    B, T, D = 1, 5, 8
    block = WalshTXLBlock(D, n_heads=2, mem_len=3)
    x = torch.randn(B, T, D)
    mem = None
    out, mem = block(x, mem)
    print("WalshTXL output:", out.shape, "mem:", mem.shape)

    # --------------------------------------------------------
    # Walsh-Perceiver IO
    # --------------------------------------------------------
    print("\n==================== WalshPerceiverIOStep ====================")
    perceiver = WalshPerceiverIOStep(d_model=8, n_heads=2, latent_len=4)
    x = torch.randn(1, 10, 8)
    q_out = torch.randn(1, 3, 8)
    y = perceiver(x, q_out)
    print("WalshPerceiver IO output:", y.shape)

    # --------------------------------------------------------
    # Walsh-Longformer
    # --------------------------------------------------------
    print("\n==================== Walsh-Longformer Sliding Window ====================")
    T, D = 10, 8
    q = torch.randn(T, D)
    k = torch.randn(T, D)
    v = torch.randn(T, D)
    w = 2
    out = walsh_sliding_window_attention(q, k, v, w)
    print("Walsh-Longformer output:", out.shape)

    # --------------------------------------------------------
    # Walsh-FlashAttention-like
    # --------------------------------------------------------
    print("\n==================== Walsh-FlashAttention-like ====================")
    T, D = 12, 8
    q = torch.randn(T, D)
    k = torch.randn(T, D)
    v = torch.randn(T, D)
    out = walsh_flash_like(q, k, v, block_size=4)
    print("Walsh-Flash-like output:", out.shape)
