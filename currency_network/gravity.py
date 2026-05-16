"""Gravitational pressure and reserve currency ranking."""

import numpy as np


def compute_gravitational_pressure(K, mass, G0_normalize=True):
    """Compute gravitational pressure (reserve currency measure)."""
    K_sq = K ** 2
    P_grav = (K_sq @ mass) * mass

    if G0_normalize:
        max_p = np.max(P_grav)
        if max_p > 1e-10:
            P_grav = P_grav / max_p

    return P_grav


def rank_reserve_currencies(P_grav, currencies, theta=2.0):
    """Rank currencies by gravitational pressure (excluding USD from threshold)."""
    # Exclude USD (index 0) and zeros when computing mean threshold
    non_zero_mask = P_grav > 0
    if non_zero_mask.sum() > 0:
        mean_p = np.mean(P_grav[non_zero_mask])
    else:
        mean_p = np.mean(P_grav)

    reserve_mask = P_grav >= theta * mean_p
    reserve_currencies = [currencies[i] for i in range(len(currencies)) if reserve_mask[i]]

    # Sort by pressure descending
    rankings = sorted(
        [(currencies[i], P_grav[i]) for i in range(len(currencies))],
        key=lambda x: x[1],
        reverse=True,
    )

    return rankings, reserve_currencies
