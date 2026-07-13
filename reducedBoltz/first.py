import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.1, seed=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr

        rng = np.random.RandomState(seed)
        # Small random weights
        self.W = 0.01 * rng.randn(n_visible, n_hidden)
        # Biases
        self.b = np.zeros(n_visible)  # visible bias
        self.c = np.zeros(n_hidden)   # hidden bias

    # ----- Core probabilities -----

    def p_h_given_v(self, v):
        # v: (batch_size, n_visible)
        return sigmoid(self.c + np.dot(v, self.W))

    def p_v_given_h(self, h):
        # h: (batch_size, n_hidden)
        return sigmoid(self.b + np.dot(h, self.W.T))

    # ----- Gibbs sampling step -----

    def sample_h_given_v(self, v):
        p_h = self.p_h_given_v(v)
        return p_h, (p_h > np.random.rand(*p_h.shape)).astype(np.float32)

    def sample_v_given_h(self, h):
        p_v = self.p_v_given_h(h)
        return p_v, (p_v > np.random.rand(*p_v.shape)).astype(np.float32)

    # ----- Contrastive Divergence (CD-1) update -----

    def cd1_update(self, v0):
        """
        v0: (batch_size, n_visible) binary data
        """
        # Positive phase
        p_h0, h0 = self.sample_h_given_v(v0)

        # Negative phase (one Gibbs step)
        p_v1, v1 = self.sample_v_given_h(h0)
        p_h1, h1 = self.sample_h_given_v(v1)

        # Compute gradients (expectations)
        # Positive: <v h^T>_data
        pos_grad = np.dot(v0.T, p_h0)
        # Negative: <v h^T>_model
        neg_grad = np.dot(v1.T, p_h1)

        batch_size = v0.shape[0]
        dW = (pos_grad - neg_grad) / batch_size
        db = np.mean(v0 - v1, axis=0)
        dc = np.mean(p_h0 - p_h1, axis=0)

        # Parameter update
        self.W += self.lr * dW
        self.b += self.lr * db
        self.c += self.lr * dc

        # Reconstruction error (for monitoring)
        recon_error = np.mean((v0 - p_v1) ** 2)
        return recon_error

    # ----- Training loop -----

    def fit(self, X, batch_size=100, n_epochs=10, verbose=True):
        """
        X: (n_samples, n_visible) binary data
        """
        n_samples = X.shape[0]
        for epoch in range(n_epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]

            epoch_err = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                v0 = X_shuffled[i:i + batch_size]
                if v0.shape[0] < batch_size:
                    continue
                err = self.cd1_update(v0)
                epoch_err += err
                n_batches += 1

            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, recon error = {epoch_err / max(n_batches,1):.6f}")

    # ----- Transform / sample -----

    def transform(self, X):
        """Return hidden probabilities given visible data."""
        return self.p_h_given_v(X)

    def reconstruct(self, X):
        """One-step reconstruction of visible units."""
        p_h = self.p_h_given_v(X)
        p_v = self.p_v_given_h(p_h)
        return p_v

if __name__ == "__main__":
    # Dummy binary data: 1000 samples, 6 visible units
    rng = np.random.RandomState(0)
    X = (rng.rand(1000, 6) > 0.5).astype(np.float32)

    rbm = RBM(n_visible=6, n_hidden=3, lr=0.1, seed=0)
    rbm.fit(X, batch_size=50, n_epochs=20)

    v = X[:5]
    recon = rbm.reconstruct(v)
    print("Original:\n", v)
    print("Reconstruction (probabilities):\n", recon)
