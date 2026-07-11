import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

x = torch.randn(2, 4, requires_grad=True)

layer = nn.Linear(4, 4)   # θ = weights + bias

h = F.relu(layer(x))      # h = activations

loss = h.sum()
loss.backward()

print("Activation h:\n", h)
print("\nGradient wrt θ (layer.weight):\n", layer.weight.grad)
print("\nGradient wrt input x:\n", x.grad)
