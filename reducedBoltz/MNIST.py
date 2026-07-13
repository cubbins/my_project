import torch
from torchvision import datasets, transforms

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float())   # binarize
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=256, lr=0.1):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(n_visible, n_hidden))
        self.b = nn.Parameter(torch.zeros(n_visible))   # visible bias
        self.c = nn.Parameter(torch.zeros(n_hidden))    # hidden bias
        self.MineTemp = nn.Parameter(torch.zeros(n_hidden))    # hidden bias

        self.lr = lr



    def sample_h(self, v):
        p_h = torch.sigmoid(self.c + v @ self.W)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        p_v = torch.sigmoid(self.b + h @ self.W.t())
        return p_v, torch.bernoulli(p_v)

    def cd1(self, v0):
        # Positive phase
        p_h0, h0 = self.sample_h(v0)

        # Negative phase
        p_v1, v1 = self.sample_v(h0)
        p_h1, h1 = self.sample_h(v1)

        # Gradients
        pos_grad = v0.t() @ p_h0
        neg_grad = v1.t() @ p_h1

        # Update
        self.W.data += self.lr * (pos_grad - neg_grad) / v0.size(0)
        self.b.data += self.lr * torch.mean(v0 - v1, dim=0)
        self.c.data += self.lr * torch.mean(p_h0 - p_h1, dim=0)

        # Reconstruction error
        return torch.mean((v0 - p_v1) ** 2)
    
rbm = RBM(n_visible=784, n_hidden=256, lr=0.1)

for epoch in range(10):
    epoch_err = 0
    for batch, _ in train_loader:
        v0 = batch.view(-1, 784)  # flatten
        err = rbm.cd1(v0)
        epoch_err += err.item()

    print(f"Epoch {epoch+1}, recon error = {epoch_err / len(train_loader):.6f}")

test_batch, _ = next(iter(train_loader))
v0 = test_batch[:8].view(-1, 784)
p_v = rbm.sample_v(rbm.sample_h(v0)[1])[0]


