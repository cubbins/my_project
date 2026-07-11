import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        # simple 2-layer MLP with residual
        h = F.relu(self.lin1(x))
        out = x + self.lin2(h)
        return out

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 4, 8
    x = torch.randn(B, T, D, requires_grad=True)

    block = TinyBlock(D)
    y = block(x)                 # forward
    loss = y.pow(2).mean()       # simple loss

    loss.backward()              # backward (backprop)

    print("Loss:", loss.item())
    print("Grad w.r.t x shape:", x.grad.shape)
    print("Grad w.r.t lin1.weight shape:", block.lin1.weight.grad.shape)
