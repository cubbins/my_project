import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0. Utility: Walsh–Hadamard Transform (FWHT)
# ============================================================

def fwht(x):
    """
    Fast Walsh–Hadamard Transform over the last dimension.
    Uses reshape() instead of view() to support non-contiguous tensors.
    """
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])  # [N, D]
    N, D = x.shape
    h = 1
    while h < D:
        x = x.reshape(N, -1, 2 * h)
        a = x[:, :, :h]
        b = x[:, :, h:2*h]
        x = torch.cat([a + b, a - b], dim=-1)
        h *= 2
    x = x.reshape(*orig_shape)
    return x / (orig_shape[-1] ** 0.5)


# ============================================================
# 1. TinyBlock (Jacobian Demo)
# ============================================================

class TinyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin1 = nn.Linear(d, d, bias=True)
        self.lin2 = nn.Linear(d, d, bias=True)

    def forward(self, x):
        z = self.lin1(x)
        h = F.relu(z)
        u = self.lin2(h)
        return x + u, z


# ============================================================
# 2. Transformer-XL Block (Segment Recurrence)
# ============================================================

class TXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, mem_len):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mem_len = mem_len

    def forward(self, x, mem):
        if mem is None:
            cat = x
        else:
            cat = torch.cat([mem.detach(), x], dim=1)

        q = x
        k = v = cat
        out, _ = self.attn(q, k, v)

        new_mem = cat[:, -self.mem_len:].detach()
        return out, new_mem


# ============================================================
# 3. Perceiver IO (Cross-Attention Bottleneck)
# ============================================================

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, q, x):
        out, _ = self.attn(q, x, x)
        return out

class PerceiverIOStep(nn.Module):
    def __init__(self, d_model, n_heads, latent_len):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_len, d_model))
        self.in2lat = CrossAttention(d_model, n_heads)
        self.lat2out = CrossAttention(d_model, n_heads)

    def forward(self, x, q_out):
        B = x.size(0)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        z = self.in2lat(lat, x)
        y = self.lat2out(q_out, z)
        return y


# ============================================================
# 4. Longformer Sliding-Window Attention
# ============================================================

def sliding_window_attention(q, k, v, w):
    T, D = q.size()
    out = torch.zeros_like(q)
    for i in range(T):
        left = max(0, i - w)
        right = min(T, i + w + 1)
        k_slice = k[left:right]
        v_slice = v[left:right]
        scores = (q[i:i+1] @ k_slice.T) / (D ** 0.5)
        attn = scores.softmax(dim=-1)
        out[i] = attn @ v_slice
    return out


# ============================================================
# 5. FlashAttention-like Streaming Softmax
# ============================================================

def naive_flash_like(q, k, v, block_size=4):
    T, D = q.size()
    o = torch.zeros_like(q)
    m = torch.full((T,), -1e9, device=q.device)
    l = torch.zeros(T, device=q.device)

    for start in range(0, T, block_size):
        end = min(T, start + block_size)
        k_blk = k[start:end]
        v_blk = v[start:end]

        scores = (q @ k_blk.T) / (D ** 0.5)

        m_new = torch.maximum(m, scores.max(dim=-1).values)
        l = torch.exp(m - m_new) * l + torch.exp(scores - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new

        weights = torch.exp(scores - m.unsqueeze(-1))
        o = o + weights @ v_blk

    o = o / l.unsqueeze(-1)
    return o


# ============================================================
# 6. WalshAttention Block
# ============================================================

class WalshAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = fwht(q)
        k = fwht(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        return out


# ============================================================
# MAIN TEST DRIVER
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    print("\n==================== 1. TinyBlock Jacobian ====================")
    D = 4
    block = TinyBlock(D)
    x = torch.randn(D, requires_grad=True)
    y, z = block(x)

    J_auto = []
    for i in range(D):
        grad_out = torch.zeros_like(y)
        grad_out[i] = 1.0
        y.backward(grad_out, retain_graph=True)
        J_auto.append(x.grad.detach().clone())
        x.grad.zero_()
    J_auto = torch.stack(J_auto, dim=0)
    print("Autograd Jacobian:\n", J_auto)

    print("\n==================== 2. Transformer-XL Block ====================")
    B, T, D = 1, 5, 8
    txl = TXLBlock(D, n_heads=2, mem_len=3)
    x = torch.randn(B, T, D)
    mem = None
    out, mem = txl(x, mem)
    print("TXL output shape:", out.shape)
    print("Updated memory shape:", mem.shape)

    print("\n==================== 3. Perceiver IO ====================")
    perceiver = PerceiverIOStep(d_model=8, n_heads=2, latent_len=4)
    x = torch.randn(1, 10, 8)
    q_out = torch.randn(1, 3, 8)
    y = perceiver(x, q_out)
    print("Perceiver IO output shape:", y.shape)

    print("\n==================== 4. Longformer Sliding Window ====================")
    T, D = 10, 8
    q = torch.randn(T, D)
    k = torch.randn(T, D)
    v = torch.randn(T, D)
    w = 2
    out = sliding_window_attention(q, k, v, w)
    print("Longformer output shape:", out.shape)

    print("\n==================== 5. FlashAttention-like ====================")
    T, D = 12, 8
    q = torch.randn(T, D)
    k = torch.randn(T, D)
    v = torch.randn(T, D)
    out = naive_flash_like(q, k, v, block_size=4)
    print("Flash-like output shape:", out.shape)

    print("\n==================== 6. WalshAttention ====================")
    B, T, D = 2, 6, 8
    walsh_attn = WalshAttention(d_model=D, n_heads=2)
    x = torch.randn(B, T, D)
    out = walsh_attn(x)
    print("WalshAttention output shape:", out.shape)
