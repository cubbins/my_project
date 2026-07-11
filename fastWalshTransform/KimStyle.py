import numpy as np

def simple_kim_smoother(alpha, P):
    """
    alpha[t, i] = P(s_t=i | y_1:t)  (filtered)
    P[i, j] = P(s_{t+1}=j | s_t=i)
    Returns gamma[t, i] = P(s_t=i | y_1:T) (smoothed)
    """
    T, K = alpha.shape
    gamma = np.zeros_like(alpha)

    # initialize with filtered at T-1
    gamma[-1] = alpha[-1]

    for t in range(T-2, -1, -1):
        for i in range(K):
            # backward recursion (simplified, no explicit likelihood term)
            gamma[t, i] = alpha[t, i] * np.sum(P[i, :] * gamma[t+1, :])
        gamma[t] /= gamma[t].sum()

    return gamma

if __name__ == "__main__":
    # toy filtered probs for T=4, K=2
    alpha = np.array([
        [0.6, 0.4],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    P = np.array([[0.9, 0.1],
                  [0.2, 0.8]])

    gamma = simple_kim_smoother(alpha, P)
    print("Filtered (alpha):")
    print(alpha)
    print("\nSmoothed (gamma):")
    print(gamma)
