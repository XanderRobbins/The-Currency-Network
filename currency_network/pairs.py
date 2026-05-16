"""Pairs trading signal and backtest."""

import numpy as np
import pandas as pd


def compute_pairs_signal(u_obs, u_star, currencies):
    """Compute pairs trading signal."""
    signals = {}

    for i in range(len(currencies)):
        for j in range(i + 1, len(currencies)):
            pair_name = f"{currencies[i]}/{currencies[j]}"
            signal = (u_obs[:, i] - u_obs[:, j]) - (u_star[:, i] - u_star[:, j])
            signals[pair_name] = signal

    return signals


def backtest_pairs_sharpe(
    u_obs, u_star, log_returns, currencies, pair=("EUR", "GBP"), bid_ask_pips=1.2, pip_size=0.0001
):
    """Backtest CN pairs strategy and naive pairs strategy."""

    # Get indices
    i = currencies.index(pair[0])
    j = currencies.index(pair[1])

    # CN signal
    cn_signal = (u_obs[:, i] - u_obs[:, j]) - (u_star[:, i] - u_star[:, j])
    signal_std = np.std(cn_signal)

    # Per paper §8.2: large positive signal → i over-displaced → SHORT i / LONG j (position = -1)
    cn_position = np.zeros(len(cn_signal))
    cn_position[cn_signal > signal_std] = -1
    cn_position[cn_signal < -signal_std] = 1

    # Compute daily PnL
    returns_i = log_returns.iloc[:, i].values[: len(cn_signal)]
    returns_j = log_returns.iloc[:, j].values[: len(cn_signal)]
    price_diff_returns = returns_i - returns_j

    # Transaction costs
    cn_pnl = np.zeros(len(cn_signal))
    cost_per_trade = bid_ask_pips * pip_size

    for t in range(1, len(cn_signal)):
        cn_pnl[t] = cn_position[t - 1] * price_diff_returns[t]
        if cn_position[t] != cn_position[t - 1]:
            cn_pnl[t] -= cost_per_trade

    # Remove startup
    cn_pnl = cn_pnl[1:]
    cn_sharpe = np.mean(cn_pnl) / (np.std(cn_pnl) + 1e-10) * np.sqrt(252)

    # Naive strategy: corr-based
    rolling_mean = pd.Series(returns_i - returns_j).rolling(window=20).mean().values
    naive_signal = (returns_i - returns_j) - rolling_mean

    naive_position = np.zeros(len(naive_signal))
    naive_std = np.std(naive_signal)
    naive_position[naive_signal > naive_std] = 1
    naive_position[naive_signal < -naive_std] = -1

    naive_pnl = np.zeros(len(naive_signal))
    for t in range(1, len(naive_signal)):
        naive_pnl[t] = naive_position[t - 1] * price_diff_returns[t]
        if naive_position[t] != naive_position[t - 1]:
            naive_pnl[t] -= cost_per_trade

    naive_pnl = naive_pnl[1:]
    naive_sharpe = np.mean(naive_pnl) / (np.std(naive_pnl) + 1e-10) * np.sqrt(252)

    return cn_sharpe, naive_sharpe, cn_pnl[1:], naive_pnl[1:]
