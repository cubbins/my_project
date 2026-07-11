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
