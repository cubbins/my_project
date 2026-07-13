import torch
from torchvision import datasets, transforms

print("Program started")

batch_size = 64

print(f"Batch size = {batch_size}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float())
])

print("Loading MNIST dataset...")

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print(f"Training samples loaded = {len(train_data)}")

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

print("DataLoader created")

import torch.nn as nn


class RBM(nn.Module):

    def __init__(self, n_visible=784, n_hidden=256, lr=0.1):

        print("\n>>> ENTERING RBM.__init__()")

        super().__init__()

        print(f"Creating weight matrix W ({n_visible} x {n_hidden})")

        self.W = nn.Parameter(
            0.01 * torch.randn(n_visible, n_hidden)
        )

        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

        self.MineTemp = nn.Parameter(torch.zeros(n_hidden))

        self.lr = lr

        print("W shape =", self.W.shape)
        print("b shape =", self.b.shape)
        print("c shape =", self.c.shape)
        print("Learning rate =", self.lr)

        print("<<< EXITING RBM.__init__()\n")


    def sample_h(self, v):

        print("\n>>> ENTERING sample_h()")
        print("Input visible shape =", v.shape)

        p_h = torch.sigmoid(self.c + v @ self.W)

        print("Hidden probability shape =", p_h.shape)

        h_sample = torch.bernoulli(p_h)

        print("Hidden sample shape =", h_sample.shape)

        print("<<< EXITING sample_h()")

        return p_h, h_sample


    def sample_v(self, h):

        print("\n>>> ENTERING sample_v()")
        print("Input hidden shape =", h.shape)

        p_v = torch.sigmoid(self.b + h @ self.W.t())

        print("Visible probability shape =", p_v.shape)

        v_sample = torch.bernoulli(p_v)

        print("Visible sample shape =", v_sample.shape)

        print("<<< EXITING sample_v()")

        return p_v, v_sample


    def cd1(self, v0):

        print("\n================================================")
        print(">>> ENTERING cd1()")
        print("v0 shape =", v0.shape)

        # Positive phase
        print("\nPositive Phase")

        p_h0, h0 = self.sample_h(v0)

        print("Mean p_h0 =", p_h0.mean().item())

        # Negative phase
        print("\nNegative Phase")

        p_v1, v1 = self.sample_v(h0)

        p_h1, h1 = self.sample_h(v1)

        # Gradients
        print("\nComputing gradients")

        pos_grad = v0.t() @ p_h0
        neg_grad = v1.t() @ p_h1

        print("pos_grad shape =", pos_grad.shape)
        print("neg_grad shape =", neg_grad.shape)

        grad_norm = torch.norm(pos_grad - neg_grad)

        print("Gradient norm =", grad_norm.item())

        # Update
        print("\nUpdating parameters")

        self.W.data += self.lr * (pos_grad - neg_grad) / v0.size(0)

        self.b.data += self.lr * torch.mean(
            v0 - v1,
            dim=0
        )

        self.c.data += self.lr * torch.mean(
            p_h0 - p_h1,
            dim=0
        )

        # Reconstruction error
        err = torch.mean((v0 - p_v1) ** 2)

        print("Reconstruction error =", err.item())

        print("<<< EXITING cd1()")
        print("================================================\n")

        return err


print("\nCreating RBM model")

rbm = RBM(
    n_visible=784,
    n_hidden=256,
    lr=0.1
)

print("RBM created successfully")

print("\nStarting training loop")

for epoch in range(10):

    print("\n################################################")
    print(f"STARTING EPOCH {epoch+1}")
    print("################################################")

    epoch_err = 0

    for batch_idx, (batch, _) in enumerate(train_loader):

        print(
            f"\nEpoch {epoch+1}, "
            f"Batch {batch_idx+1}/{len(train_loader)}"
        )

        v0 = batch.view(-1, 784)

        print("Flattened batch shape =", v0.shape)

        err = rbm.cd1(v0)

        epoch_err += err.item()

        if batch_idx % 100 == 0:
            print(
                f"Running average error = "
                f"{epoch_err/(batch_idx+1):.6f}"
            )

    avg_err = epoch_err / len(train_loader)

    print("\nEpoch completed")
    print(f"Epoch {epoch+1}, recon error = {avg_err:.6f}")

print("\nTraining complete")

print("\nGenerating reconstruction example")

test_batch, _ = next(iter(train_loader))

print("Test batch shape =", test_batch.shape)

v0 = test_batch[:8].view(-1, 784)

print("Selected 8 images")
print("v0 shape =", v0.shape)

print("\nSampling hidden layer")

hidden_probs, hidden_states = rbm.sample_h(v0)

print("Sampling reconstructed visible layer")

p_v, reconstructed = rbm.sample_v(hidden_states)

print("Reconstruction shape =", p_v.shape)

print("\nProgram finished")