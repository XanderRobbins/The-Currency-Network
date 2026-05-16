"""Core mathematical models: Laplacian, mass, equilibrium."""

import numpy as np
from sklearn.covariance import LedoitWolf


def compute_correlation_matrix(log_returns_window, shrinkage=True):
    """Compute correlation matrix with optional shrinkage."""
    if shrinkage:
        lw = LedoitWolf()
        cov = lw.fit(log_returns_window).covariance_
        # Convert covariance to correlation
        D = np.diag(1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-10)))
        corr = D @ cov @ D
    else:
        corr = np.corrcoef(log_returns_window.T)

    # Clip to [-1, 1] and set diagonal to 0
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 0)

    return corr


def compute_spring_constants(corr_matrix, k0_normalize=True):
    """Compute spring constants and sign matrix from correlation matrix."""
    N = corr_matrix.shape[0]
    K_raw = np.abs(corr_matrix)
    alpha = np.sign(corr_matrix)

    if k0_normalize:
        frobenius_norm = np.linalg.norm(K_raw, "fro")
        if frobenius_norm > 1e-10:
            k0 = N / frobenius_norm
        else:
            k0 = 1.0
        K = k0 * K_raw
    else:
        k0 = 1.0
        K = K_raw

    return K, alpha, k0


def compute_signed_laplacian(K, alpha):
    """Compute signed Laplacian from spring constants and sign matrix."""
    N = K.shape[0]
    L = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                L[i, j] = -alpha[i, j] * K[i, j]
        L[i, i] = np.sum(K[i, :])

    return L


def compute_mass(log_returns_window, volume_window, beta=(0.5, 0.3, 0.2)):
    """Compute mass vector from log returns, volume, and volatility."""
    N = log_returns_window.shape[1]
    beta1, beta2, beta3 = beta

    # Volume component (already normalized to [0,1])
    V = volume_window.mean(axis=0)
    V = np.clip(V, 1e-6, 1.0)

    # Volatility component (inversion: low vol -> high mass)
    sigma = log_returns_window.std(axis=0)
    sigma_z = (sigma - sigma.mean()) / (sigma.std() + 1e-10)
    Omega = 1.0 / (1.0 + sigma_z)
    Omega = (Omega - Omega.min()) / (Omega.max() - Omega.min() + 1e-10) + 1e-6

    # Information ratio component
    mu = log_returns_window.mean(axis=0)
    IR = np.abs(mu) / (sigma + 1e-8)
    IR = (IR - IR.min()) / (IR.max() - IR.min() + 1e-10) + 1e-6

    # Combine
    mass = (V ** beta1) * (Omega ** beta2) * (IR ** beta3)

    return mass


def compute_equilibrium(L, f):
    """Compute equilibrium displacement u* = L_pinv @ f."""
    L_plus = np.linalg.pinv(L)
    u_star = L_plus @ f
    return u_star, L_plus


def compute_laplacian_eigenvalues(L):
    """Compute eigenvalues of Laplacian."""
    eigenvalues = np.linalg.eigvalsh(L)
    return np.sort(eigenvalues)


def compute_fiedler_value(L):
    """Compute Fiedler value (second smallest eigenvalue)."""
    eigs = compute_laplacian_eigenvalues(L)
    return eigs[1]


def compute_displacement(log_returns, tau=60):
    """
    u_obs_i(t) = log_return_i(t) minus the rolling tau-day mean return of currency i.
    This is the daily deviation from trend — a stationary quantity.
    The CN model's equilibrium u* predicts this daily displacement.
    """
    rolling_mean = log_returns.rolling(window=tau, min_periods=tau//2).mean()
    u_obs = log_returns - rolling_mean
    return u_obs.dropna()
