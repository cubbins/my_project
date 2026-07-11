import numpy as np
from numpy.random import default_rng
rng = default_rng(0)

###############################################
# 1. Simulate Markov-switching AR(1) data
###############################################

def simulate_ms_ar1(T=200):
    # Parameters
    mu = np.array([0.0, 3.0])      # regime means
    phi = 0.8                      # AR coefficient
    sigma = 1.0                    # noise std

    P = np.array([[0.95, 0.05],
                  [0.10, 0.90]])   # transition matrix

    s = np.zeros(T, dtype=int)
    y = np.zeros(T)

    # initial state
    s[0] = rng.choice([0,1])
    y[0] = mu[s[0]] + rng.normal(scale=sigma)

    for t in range(1, T):
        s[t] = rng.choice([0,1], p=P[s[t-1]])
        y[t] = mu[s[t]] + phi * y[t-1] + rng.normal(scale=sigma)

    return y, s, P, mu, phi, sigma


###############################################
# 2. Hamilton Filter (Forward Probabilities)
###############################################

def hamilton_filter(y, P, mu, phi, sigma):
    T = len(y)
    K = len(mu)  # number of regimes

    # filtered probabilities α_t(j) = P(s_t=j | y_1:t)
    alpha = np.zeros((T, K))

    # initial probabilities (uniform)
    alpha[0] = np.ones(K) / K

    # likelihood function for each regime
    def likelihood(y_t, y_prev, j):
        mean = mu[j] + phi * y_prev
        return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(y_t - mean)**2/(2*sigma**2))

    for t in range(1, T):
        for j in range(K):
            # prediction step: sum over previous states
            pred = np.sum(alpha[t-1] * P[:, j])
            # update step: multiply by likelihood
            alpha[t, j] = pred * likelihood(y[t], y[t-1], j)

        # normalize
        alpha[t] /= np.sum(alpha[t])

    return alpha


###############################################
# 3. Kim Smoother (Backward Smoothing)
###############################################

def kim_smoother(alpha, P, y, mu, phi, sigma):
    T, K = alpha.shape
    smoothed = np.zeros_like(alpha)

    # initialize with filtered probabilities
    smoothed[-1] = alpha[-1]

    # likelihood function
    def likelihood(y_t, y_prev, j):
        mean = mu[j] + phi * y_prev
        return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(y_t - mean)**2/(2*sigma**2))

    # backward recursion
    for t in range(T-2, -1, -1):
        for i in range(K):
            # numerator: sum over next states
            num = 0.0
            for j in range(K):
                num += P[i, j] * likelihood(y[t+1], y[t], j) * smoothed[t+1, j]

            smoothed[t, i] = alpha[t, i] * num

        # normalize
        smoothed[t] /= np.sum(smoothed[t])

    return smoothed


###############################################
# 4. Test the full pipeline
###############################################

if __name__ == "__main__":
    y, true_states, P, mu, phi, sigma = simulate_ms_ar1(T=200)

    alpha = hamilton_filter(y, P, mu, phi, sigma)
    smoothed = kim_smoother(alpha, P, y, mu, phi, sigma)

    print("Filtered probabilities (last 5 rows):")
    print(alpha[-5:])
    print("\nSmoothed probabilities (last 5 rows):")
    print(smoothed[-5:])
    print("\nTrue states (last 10):")
    print(true_states[-10:])
