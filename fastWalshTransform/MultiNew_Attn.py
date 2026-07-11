



import torch
import torch.nn as nn
import torch.nn.functional as F

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

D = 4
torch.manual_seed(0)
block = TinyBlock(D)

x = torch.randn(D, requires_grad=True)
y, z = block(x)

# Analytical Jacobian via autograd
J_auto = []
for i in range(D):
    grad_out = torch.zeros_like(y)
    grad_out[i] = 1.0
    y.backward(grad_out, retain_graph=True)
    J_auto.append(x.grad.detach().clone())
    x.grad.zero_()
J_auto = torch.stack(J_auto, dim=0)  # D x D

print("Autograd Jacobian:\n", J_auto)





import torch
import torch.nn as nn
import torch.nn.functional as F

class TXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, mem_len):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mem_len = mem_len

    def forward(self, x, mem):
        # x: [B, T, D], mem: [B, M, D] or None
        if mem is None:
            cat = x
        else:
            cat = torch.cat([mem.detach(), x], dim=1)  # [B, M+T, D]

        q = x
        k = v = cat
        out, _ = self.attn(q, k, v)  # segment-level recurrence

        # update memory
        new_mem = cat[:, -self.mem_len:].detach()
        return out, new_mem







class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, q, x):
        # q: [B, L, D] (latent or output queries)
        # x: [B, N, D] (inputs or latents)
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
        z = self.in2lat(lat, x)      # input -> latent
        y = self.lat2out(q_out, z)   # latent -> output
        return y






def sliding_window_attention(q, k, v, w):
    # q,k,v: [T, D] (single sequence, single head for simplicity)
    T, D = q.size()
    out = torch.zeros_like(q)
    for i in range(T):
        left = max(0, i - w)
        right = min(T, i + w + 1)
        k_slice = k[left:right]          # [L, D]
        v_slice = v[left:right]          # [L, D]
        scores = (q[i:i+1] @ k_slice.T) / (D ** 0.5)  # [1, L]
        attn = scores.softmax(dim=-1)    # [1, L]
        out[i] = attn @ v_slice          # [1, D] -> [D]
    return out







def naive_flash_like(q, k, v, block_size=64):
    # q,k,v: [T, D]
    T, D = q.size()
    o = torch.zeros_like(q)
    m = torch.full((T,), -1e9, device=q.device)
    l = torch.zeros(T, device=q.device)

    for start in range(0, T, block_size):
        end = min(T, start + block_size)
        k_blk = k[start:end]          # [B, D]
        v_blk = v[start:end]          # [B, D]

        scores = (q @ k_blk.T) / (D ** 0.5)  # [T, B]

        m_new = torch.maximum(m, scores.max(dim=-1).values)
        # rescale old l, add new exp
        l = torch.exp(m - m_new) * l + torch.exp(scores - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new

        # accumulate output
        weights = torch.exp(scores - m.unsqueeze(-1))  # [T, B]
        o = o + weights @ v_blk

    o = o / l.unsqueeze(-1)
    return o







