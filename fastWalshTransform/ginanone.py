import numpy as np

np.random.seed(0)

###############################################
# 1. Simulate dynamic factor + GDP
###############################################

def simulate_data(T_months=60, n_series=5):
    """
    T_months: number of months
    n_series: number of monthly indicators
    Returns:
        x: (T_months, n_series) monthly indicators
        gdp_q: (T_quarters,) quarterly GDP (every 3rd month)
        factor: (T_months,) latent factor
    """
    phi = 0.8          # AR(1) for factor
    sigma_f = 0.5      # factor shock
    sigma_e = 0.3      # idiosyncratic noise
    sigma_g = 0.4      # GDP noise

    # factor loadings for monthly indicators
    lam = np.linspace(0.5, 1.5, n_series)

    T = T_months
    f = np.zeros(T)
    x = np.zeros((T, n_series))

    # simulate factor
    f[0] = np.random.normal()
    for t in range(1, T):
        f[t] = phi * f[t-1] + np.random.normal(scale=sigma_f)

    # simulate monthly indicators
    for t in range(T):
        x[t] = lam * f[t] + np.random.normal(scale=sigma_e, size=n_series)

    # simulate quarterly GDP as factor at last month of quarter + noise
    T_q = T // 3
    gdp_q = np.zeros(T_q)
    for q in range(T_q):
        t_q = (q+1)*3 - 1  # month index (2,5,8,...)
        gdp_q[q] = 1.2 * f[t_q] + np.random.normal(scale=sigma_g)

    return x, gdp_q, f, lam, phi, sigma_f, sigma_e, sigma_g


###############################################
# 2. Kalman filter for 1-factor model
###############################################

def kalman_filter_factor22(x, lam, phi, sigma_f, sigma_e):
    """
    Simple 1-factor dynamic factor model:
        f_t = phi f_{t-1} + eta_t
        x_t = lam f_t + e_t
    x: (T, n_series)
    lam: (n_series,)
    Returns:
        f_filt: (T,) filtered factor estimates
    """
    T, n = x.shape

    # State: f_t (scalar)
    # Transition: f_t = phi f_{t-1} + eta_t, Var(eta_t) = sigma_f^2
    # Measurement: x_t = lam f_t + e_t, Var(e_t) = sigma_e^2 I

    f_pred = 0.0
    P_pred = 1.0  # initial variance

    f_filt = np.zeros(T)
    P_filt = np.zeros(T)

    for t in range(T):
        # Prediction already in f_pred, P_pred

        # Handle missing indicators (NaN) by selecting observed ones
        obs_mask = ~np.isnan(x[t])
        lam_t = lam[obs_mask]
        x_t = x[t, obs_mask]

        if len(lam_t) > 0:
            # Measurement variance: sigma_e^2 I
            R_t = (sigma_e**2) * np.eye(len(lam_t))

            # Measurement matrix: H_t = lam_t^T (row vector)
            H_t = lam_t.reshape(1, -1)  # 1 x m
            H_t_T = H_t.T               # m x 1

            # Predicted measurement mean and variance
            y_pred = (lam_t * f_pred)   # m-dim
            # Innovation
            v_t = x_t - y_pred

            # Innovation variance:
            S_t = H_t @ np.array([[P_pred]]) @ H_t_T + R_t  # m x m

            # Kalman gain (scalar state, m obs)
            # K = P_pred * H^T * S^{-1}
            S_inv = np.linalg.inv(S_t)
            K_t = P_pred * (H_t_T @ S_inv)  # m x 1

            # Update state
            f_upd = f_pred + (K_t.T @ v_t)[0]

            # Update variance
            P_upd = P_pred - (K_t.T @ H_t @ np.array([[P_pred]]))[0,0]
        else:
            # No observation: prediction = update
            f_upd = f_pred
            P_upd = P_pred + sigma_f**2  # just evolve variance

        f_filt[t] = f_upd
        P_filt[t] = P_upd

        # Predict next
        f_pred = phi * f_upd
        P_pred = phi**2 * P_upd + sigma_f**2

    return f_filt, P_filt


def kalman_filter_factor(x, lam, phi, sigma_f, sigma_e):
    """
    1-factor dynamic factor model:
        f_t = phi f_{t-1} + eta_t
        x_t = lam f_t + e_t
    Handles missing data (NaN) correctly.
    """
    T, n = x.shape

    f_pred = 0.0
    P_pred = 1.0

    f_filt = np.zeros(T)
    P_filt = np.zeros(T)

    for t in range(T):

        # Which indicators are observed?
        obs_mask = ~np.isnan(x[t])
        lam_t = lam[obs_mask]              # shape (m,)
        x_t = x[t, obs_mask]               # shape (m,)
        m = len(lam_t)

        # If we have observations:
        if m > 0:
            # H_t is m×1
            H_t = lam_t.reshape(m, 1)

            # R_t is m×m
            R_t = (sigma_e**2) * np.eye(m)

            # Innovation: v_t = x_t - H_t f_pred
            v_t = x_t - (H_t[:, 0] * f_pred)

            # Innovation variance: S_t = H P H' + R
            S_t = H_t @ np.array([[P_pred]]) @ H_t.T + R_t

            # Kalman gain: K = P_pred H' S^{-1}   (shape m×1)
            K_t = P_pred * (H_t.T @ np.linalg.inv(S_t))  # shape (1×m)

            # Update state
            f_upd = f_pred + (K_t @ v_t)[0]

            # Update variance
            P_upd = P_pred - (K_t @ H_t @ np.array([[P_pred]]))[0, 0]

        else:
            # No data → prediction = update
            f_upd = f_pred
            P_upd = P_pred + sigma_f**2

        # Store
        f_filt[t] = f_upd
        P_filt[t] = P_upd

        # Predict next
        f_pred = phi * f_upd
        P_pred = phi**2 * P_upd + sigma_f**2

    return f_filt, P_filt












###############################################
# 3. Nowcasting quarterly GDP from monthly factor
###############################################

def nowcast_gdp22(x, gdp_q, lam, phi, sigma_f, sigma_e, sigma_g):
    """
    For each quarter q, use data available up to the last month of that quarter
    to nowcast GDP_q via factor estimate.
    """
    T, n = x.shape
    T_q = len(gdp_q)

    # Introduce a "jagged edge": randomly drop some monthly indicators
    x_obs = x.copy()
    mask = np.random.rand(*x_obs.shape) < 0.1  # 10% missing
    x_obs[mask] = np.nan

    gdp_nowcast = np.zeros(T_q)

    for q in range(T_q):
        t_q = (q+1)*3 - 1  # last month of quarter
        # Use data up to t_q
        f_filt, P_filt = kalman_filter_factor(x_obs[:t_q+1], lam, phi, sigma_f, sigma_e)
        f_hat_tq = f_filt[t_q]

        # Simple GDP measurement: y_q = 1.2 f_tq + noise
        gdp_nowcast[q] = 1.2 * f_hat_tq  # conditional mean

    return gdp_nowcast, x_obs


def nowcast_gdp(x, gdp_q, lam, phi, sigma_f, sigma_e, sigma_g):
    """
    For each quarter q, use data up to last month of that quarter
    to nowcast GDP_q via filtered factor.
    """
    T, n = x.shape
    T_q = len(gdp_q)

    # Introduce missing data (jagged edge)
    x_obs = x.copy()
    mask = np.random.rand(*x_obs.shape) < 0.1
    x_obs[mask] = np.nan

    gdp_nowcast = np.zeros(T_q)

    for q in range(T_q):
        t_q = (q+1)*3 - 1  # last month of quarter

        # Filter factor using data up to t_q
        f_filt, P_filt = kalman_filter_factor(
            x_obs[:t_q+1], lam, phi, sigma_f, sigma_e
        )

        f_hat = f_filt[t_q]

        # GDP measurement: y_q = 1.2 f_tq + noise
        gdp_nowcast[q] = 1.2 * f_hat

    return gdp_nowcast, x_obs











###############################################
# 4. Main test
###############################################

if __name__ == "__main__":
    x, gdp_q, f_true, lam, phi, sigma_f, sigma_e, sigma_g = simulate_data(T_months=60, n_series=5)

    gdp_nowcast, x_obs = nowcast_gdp(x, gdp_q, lam, phi, sigma_f, sigma_e, sigma_g)

    print("True quarterly GDP (first 8):")
    print(np.round(gdp_q[:8], 3))
    print("\nNowcasted quarterly GDP (first 8):")
    print(np.round(gdp_nowcast[:8], 3))
