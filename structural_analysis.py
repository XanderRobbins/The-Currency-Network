"""
Structural Analysis: Three robustness tests for the Currency Network model.

Problem 1 — Lagged forcing function (is u* a genuine forecast?)
Problem 2 — Bootstrap stability of gravitational pressure rankings
Problem 3 — Separate CNSI from realized volatility
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import pinv as scipy_pinv
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.ar_model import AutoReg

np.random.seed(42)

# ── Data loading ────────────────────────────────────────────────────────────────

from currency_network.data import download_data, compute_log_returns, compute_volume_proxy, NETWORK_CURRENCIES
from currency_network.model import (
    compute_correlation_matrix, compute_spring_constants, compute_signed_laplacian,
    compute_equilibrium, compute_fiedler_value, compute_displacement, compute_mass,
)
from currency_network.gravity import compute_gravitational_pressure, rank_reserve_currencies
from currency_network.cnsi import CRISIS_WINDOWS

TAU = 60
BETA = (0.47, 0.31, 0.22)


def load_data():
    prices = download_data()
    log_returns = compute_log_returns(prices)
    volume_proxy = compute_volume_proxy(log_returns)
    return prices, log_returns, volume_proxy


# ════════════════════════════════════════════════════════════════════════════════
# PROBLEM 1: LAGGED FORCING FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def run_lagged_backtest(log_returns, volume_proxy, tau=TAU, lag=1):
    """
    Like run_rolling_backtest but f(t) = u_obs(t - lag) - mean instead of u_obs(t).
    Returns dates, u_star, u_obs arrays.
    """
    log_ret_net = log_returns.iloc[:, 1:]
    vol_net = volume_proxy.iloc[:, 1:]
    displacement_all = compute_displacement(log_ret_net, tau)

    mask = (log_returns.index >= "2015-01-01") & (log_returns.index <= "2024-12-31")
    dates = log_returns.index[mask]

    results = {"dates": [], "u_star": [], "u_obs": [], "fiedler": [], "L_series": [], "f_series": []}

    from tqdm import tqdm
    for date in tqdm(dates, desc=f"Lagged backtest (lag={lag})"):
        date_loc = log_returns.index.get_loc(date)
        if date_loc < tau + lag:
            continue

        window_start = date_loc - tau
        window_end = date_loc
        log_ret_window = log_ret_net.iloc[window_start:window_end].values
        vol_window = vol_net.iloc[window_start:window_end].values

        if np.isnan(log_ret_window).sum() > (tau * log_ret_window.shape[1] / 2):
            continue

        # Forcing at date_loc - lag (fully observable at t)
        lag_date = log_returns.index[date_loc - lag]
        if lag_date not in displacement_all.index:
            continue
        u_obs_lag = displacement_all.loc[lag_date].values
        if np.isnan(u_obs_lag).any():
            continue

        # Current u_obs (target)
        if date not in displacement_all.index:
            continue
        u_obs_now = displacement_all.loc[date].values
        if np.isnan(u_obs_now).any():
            continue

        corr = compute_correlation_matrix(log_ret_window, shrinkage=True)
        K, alpha, k0 = compute_spring_constants(corr)
        L = compute_signed_laplacian(K, alpha)

        f = u_obs_lag.copy()
        f = f - f.mean()  # enforce sum-to-zero

        u_star, _ = compute_equilibrium(L, f)
        fiedler = compute_fiedler_value(L)

        results["dates"].append(date)
        results["u_star"].append(u_star)
        results["u_obs"].append(u_obs_now)
        results["fiedler"].append(fiedler)
        results["L_series"].append(L)
        results["f_series"].append(f)

    results["u_star"] = np.array(results["u_star"])
    results["u_obs"] = np.array(results["u_obs"])
    results["fiedler"] = np.array(results["fiedler"])
    results["f_series"] = np.array(results["f_series"])
    results["dates"] = np.array(results["dates"])
    return results


def compute_r2_lagged(u_star, u_obs, currencies):
    """
    u_star[t] (built from f at t-1) predicts u_obs[t+1].
    Shift by 1: u_star[:-1] vs u_obs[1:].
    Also report in-sample vs OOS.
    """
    u_star_pred = u_star[:-1]
    u_obs_target = u_obs[1:]

    r2_raw = {}
    for i, curr in enumerate(currencies):
        ss_res = np.sum((u_obs_target[:, i] - u_star_pred[:, i]) ** 2)
        ss_tot = np.sum((u_obs_target[:, i] - u_obs_target[:, i].mean()) ** 2)
        r2_raw[curr] = max(1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0, -1.0)

    return r2_raw


def ar1_oos_benchmark(u_obs, dates, currencies, train_end="2019-12-31"):
    """AR(1) fit on 2015-2019, predict 2020-2024 out-of-sample."""
    train_mask = dates[:-1] <= pd.Timestamp(train_end)
    test_mask = dates[:-1] > pd.Timestamp(train_end)

    r2_dict = {}
    for i, curr in enumerate(currencies):
        u_i = u_obs[:, i]
        train_y = u_i[:-1][train_mask]
        train_x = u_i[1:][train_mask]  # AR(1): predict u[t+1] from u[t]
        # Fit simple OLS AR(1)
        if len(train_y) < 10:
            r2_dict[curr] = 0.0
            continue
        beta_ar = np.cov(train_y, train_x)[0, 1] / (np.var(train_y) + 1e-12)
        alpha_ar = np.mean(train_x) - beta_ar * np.mean(train_y)

        # Evaluate OOS
        test_x_pred = alpha_ar + beta_ar * u_i[:-1][test_mask]
        test_x_actual = u_i[1:][test_mask]
        if len(test_x_actual) == 0:
            r2_dict[curr] = 0.0
            continue
        ss_res = np.sum((test_x_actual - test_x_pred) ** 2)
        ss_tot = np.sum((test_x_actual - test_x_actual.mean()) ** 2)
        r2_dict[curr] = max(1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0, -1.0)

    return r2_dict


def compute_cnsi_lagged(backtest_results):
    """Compute CNSI from lagged backtest results."""
    cnsi_list = []
    for i in range(len(backtest_results["u_star"])):
        u_star = backtest_results["u_star"][i]
        L = backtest_results["L_series"][i]
        cnsi = 0.5 * u_star @ L @ u_star
        cnsi_list.append(cnsi)
    return np.array(cnsi_list)


def detect_crisis_spikes_series(cnsi_series, dates, window=252, threshold=2.5):
    rolling_mean = pd.Series(cnsi_series).rolling(window=window, min_periods=window // 2).mean().values
    rolling_std = pd.Series(cnsi_series).rolling(window=window, min_periods=window // 2).std().values
    z_score = (cnsi_series - rolling_mean) / (rolling_std + 1e-10)
    return z_score


def cnsi_event_zscores(cnsi_z, dates, events):
    """Report max z-score for each event window."""
    out = {}
    for event_name, (start, end) in events.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if not mask.any():
            out[event_name] = np.nan
            continue
        z_in_window = cnsi_z[mask]
        valid = ~np.isnan(z_in_window)
        out[event_name] = float(np.max(z_in_window[valid])) if valid.any() else np.nan
    return out


def problem1(log_returns, volume_proxy):
    print("\n" + "=" * 70)
    print("PROBLEM 1: LAGGED FORCING FUNCTION")
    print("=" * 70)

    # Option 1: f(t) = u_obs(t-1), u*(t) predicts u_obs(t+1)
    print("\n--- Option 1: f(t) = u_obs(t-1) [1-day lagged displacement] ---")
    res_lag = run_lagged_backtest(log_returns, volume_proxy, tau=TAU, lag=1)

    dates_lag = res_lag["dates"]
    u_star_lag = res_lag["u_star"]
    u_obs_lag_arr = res_lag["u_obs"]

    # R² for u*(t) predicting u_obs(t+1), OOS only (2020-2024)
    oos_mask = dates_lag >= pd.Timestamp("2020-01-01")
    u_star_oos = u_star_lag[oos_mask]
    u_obs_oos = u_obs_lag_arr[oos_mask]

    print(f"\n  OOS window: {dates_lag[oos_mask][0].date()} to {dates_lag[oos_mask][-1].date()} ({oos_mask.sum()} days)")

    r2_lagged_oos = compute_r2_lagged(u_star_oos, u_obs_oos, NETWORK_CURRENCIES)
    print("\n  R²: u*(t) [built from f=u_obs(t-1)] predicting u_obs(t+1), 2020-2024 OOS:")
    print(f"  {'Currency':<8} {'R²':>8}")
    print(f"  {'-'*18}")
    for curr in NETWORK_CURRENCIES:
        print(f"  {curr:<8} {r2_lagged_oos[curr]:>8.4f}")
    avg_r2_lagged = np.mean(list(r2_lagged_oos.values()))
    print(f"  {'Average':<8} {avg_r2_lagged:>8.4f}")

    # AR(1) strict OOS comparison (fit 2015-2019, eval 2020-2024)
    r2_ar1_oos = ar1_oos_benchmark(u_obs_lag_arr, dates_lag, NETWORK_CURRENCIES)
    avg_r2_ar1 = np.mean(list(r2_ar1_oos.values()))
    print(f"\n  AR(1) strict OOS (fit 2015-2019, eval 2020-2024):")
    print(f"  {'Currency':<8} {'AR(1) R²':>10}")
    print(f"  {'-'*20}")
    for curr in NETWORK_CURRENCIES:
        print(f"  {curr:<8} {r2_ar1_oos[curr]:>10.4f}")
    print(f"  {'Average':<8} {avg_r2_ar1:>10.4f}")

    # CNSI with lagged f — do crisis detections survive?
    print("\n--- CNSI computed with lagged f(t) = u_obs(t-1) ---")
    cnsi_lag = compute_cnsi_lagged(res_lag)
    cnsi_z_lag = detect_crisis_spikes_series(cnsi_lag, dates_lag)

    target_events = {
        "Brexit (Jun 2016)": ("2016-06-20", "2016-07-31"),
        "COVID (Mar 2020)": ("2020-02-01", "2020-04-30"),
        "Japan Carry Unwind (Aug 2024)": ("2024-07-25", "2024-09-15"),
    }
    control_events = {
        "Russia/Ukraine (Feb 2022)": ("2022-01-15", "2022-04-30"),
        "SVB Crisis (Mar 2023)": ("2023-03-01", "2023-04-30"),
        "US Election (Nov 2016)": ("2016-11-07", "2016-11-30"),
        "Truss/Yen Crisis (Sep 2022)": ("2022-08-01", "2022-11-30"),
    }

    z_target = cnsi_event_zscores(cnsi_z_lag, dates_lag, target_events)
    z_control = cnsi_event_zscores(cnsi_z_lag, dates_lag, control_events)

    print("\n  True positive events (should detect at z > 2.5):")
    print(f"  {'Event':<40} {'Max z':>8} {'Detected':>10}")
    print(f"  {'-'*60}")
    for ev, z in z_target.items():
        det = "YES" if (not np.isnan(z) and z > 2.5) else "NO"
        print(f"  {ev:<40} {z:>8.2f} {det:>10}")

    print("\n  Correct-negative events (should NOT detect at z > 2.5):")
    print(f"  {'Event':<40} {'Max z':>8} {'Fired':>10}")
    print(f"  {'-'*60}")
    for ev, z in z_control.items():
        fired = "YES (FP)" if (not np.isnan(z) and z > 2.5) else "NO"
        print(f"  {ev:<40} {z:>8.2f} {fired:>10}")

    return res_lag, cnsi_z_lag, dates_lag


# ════════════════════════════════════════════════════════════════════════════════
# PROBLEM 2: BOOTSTRAP GRAVITATIONAL PRESSURE RANKINGS
# ════════════════════════════════════════════════════════════════════════════════

def compute_gravity_for_window(log_ret_window, vol_window, beta=BETA):
    """Compute gravitational pressure ranking for a single 252-day window."""
    if np.isnan(log_ret_window).sum() > (log_ret_window.shape[0] * log_ret_window.shape[1] / 2):
        return None
    corr = compute_correlation_matrix(log_ret_window, shrinkage=True)
    K, alpha, _ = compute_spring_constants(corr)
    L = compute_signed_laplacian(K, alpha)
    K_adj = np.abs(L - np.diag(np.diag(L)))
    mass = compute_mass(log_ret_window, vol_window, beta=beta)
    P_grav = compute_gravitational_pressure(K_adj, mass, G0_normalize=True)
    # Return rank (1=highest) for each currency
    order = np.argsort(-P_grav)
    ranks = np.empty(len(P_grav), dtype=int)
    for rank_pos, idx in enumerate(order):
        ranks[idx] = rank_pos + 1
    return ranks


def problem2(log_returns, volume_proxy, n_bootstrap=500, window=252):
    print("\n" + "=" * 70)
    print("PROBLEM 2: BOOTSTRAP GRAVITATIONAL PRESSURE RANKINGS")
    print("=" * 70)

    log_ret_net = log_returns.iloc[:, 1:]
    vol_net = volume_proxy.iloc[:, 1:]

    # All valid start indices for 252-day blocks
    n_total = len(log_ret_net)
    max_start = n_total - window
    valid_starts = np.arange(0, max_start + 1)

    print(f"\n  Total trading days: {n_total}")
    print(f"  Block window: {window} days")
    print(f"  Bootstrap samples: {n_bootstrap}")
    print(f"  Valid block starts: {len(valid_starts)}")

    all_ranks = []  # shape: (n_bootstrap, 9)
    np.random.seed(42)

    from tqdm import tqdm
    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        start = np.random.choice(valid_starts)
        end = start + window

        log_ret_window = log_ret_net.iloc[start:end].values
        vol_window = vol_net.iloc[start:end].values

        ranks = compute_gravity_for_window(log_ret_window, vol_window)
        if ranks is not None:
            all_ranks.append(ranks)

    all_ranks = np.array(all_ranks)  # (n_valid, 9)
    print(f"\n  Successful bootstrap draws: {len(all_ranks)}")

    print(f"\n  {'Currency':<8} {'Median Rank':>12} {'p5 Rank':>10} {'p95 Rank':>10} {'Frac Top3':>12}")
    print(f"  {'-'*56}")

    # Sort by median rank to show canonical order
    median_ranks = np.median(all_ranks, axis=0)
    sort_order = np.argsort(median_ranks)

    for idx in sort_order:
        curr = NETWORK_CURRENCIES[idx]
        med = np.median(all_ranks[:, idx])
        p5 = np.percentile(all_ranks[:, idx], 5)
        p95 = np.percentile(all_ranks[:, idx], 95)
        frac_top3 = np.mean(all_ranks[:, idx] <= 3)
        print(f"  {curr:<8} {med:>12.1f} {p5:>10.1f} {p95:>10.1f} {frac_top3:>12.1%}")

    # EUR #1 fraction
    eur_idx = NETWORK_CURRENCIES.index("EUR")
    eur_frac_1 = np.mean(all_ranks[:, eur_idx] == 1)
    print(f"\n  EUR is ranked #1 in {eur_frac_1:.1%} of bootstrap draws.")

    # Rank stability: mean rank IQR across all currencies
    rank_iqrs = [np.percentile(all_ranks[:, i], 75) - np.percentile(all_ranks[:, i], 25)
                 for i in range(all_ranks.shape[1])]
    print(f"  Mean rank IQR across all currencies: {np.mean(rank_iqrs):.2f} (lower = more stable)")

    return all_ranks


# ════════════════════════════════════════════════════════════════════════════════
# PROBLEM 3: SEPARATE CNSI FROM REALIZED VOLATILITY
# ════════════════════════════════════════════════════════════════════════════════

def compute_rvol(log_returns, window=252):
    """
    Cross-sectional daily realized volatility: RVol(t) = mean(|r_i(t)|) across 9 currencies.
    RVol-Z: same rolling z-score as CNSI (252-day trailing, lagged 1 day).
    """
    log_ret_net = log_returns.iloc[:, 1:]  # 9 non-USD

    # Daily cross-sectional mean of |r_i|
    rvol = log_ret_net.abs().mean(axis=1)

    # Rolling z-score with same window as CNSI
    rolling_mean = rvol.rolling(window=window, min_periods=window // 2).mean().shift(1)
    rolling_std = rvol.rolling(window=window, min_periods=window // 2).std().shift(1)
    rvol_z = (rvol - rolling_mean) / (rolling_std + 1e-10)

    return rvol, rvol_z


def problem3(log_returns, cnsi_z_lag, dates_lag):
    """
    Compare CNSI-Z (lagged f version) vs RVol-Z.
    Also load original CNSI-Z from saved results for comparison.
    """
    print("\n" + "=" * 70)
    print("PROBLEM 3: CNSI vs REALIZED VOLATILITY")
    print("=" * 70)

    rvol, rvol_z = compute_rvol(log_returns)

    # Align RVol-Z to backtest dates
    rvol_z_aligned = rvol_z.reindex(pd.DatetimeIndex(dates_lag))

    cnsi_z_series = pd.Series(cnsi_z_lag, index=pd.DatetimeIndex(dates_lag))

    # Drop NaN (start of series)
    valid = ~(cnsi_z_series.isna() | rvol_z_aligned.isna())
    cnsi_clean = cnsi_z_series[valid]
    rvol_clean = rvol_z_aligned[valid]

    # Full-sample correlation
    corr_full = cnsi_clean.corr(rvol_clean)
    print(f"\n  Full-sample correlation(CNSI-Z, RVol-Z): {corr_full:.4f}")

    # Per-event comparison
    all_events = {
        "Brexit (Jun 2016)":          ("2016-06-20", "2016-07-31"),
        "COVID (Mar 2020)":           ("2020-02-01", "2020-04-30"),
        "Japan Carry Unwind (Aug 2024)": ("2024-07-25", "2024-09-15"),
        "Russia/Ukraine (Feb 2022)":  ("2022-01-15", "2022-04-30"),
        "SVB Crisis (Mar 2023)":      ("2023-03-01", "2023-04-30"),
        "US Election (Nov 2016)":     ("2016-11-07", "2016-11-30"),
        "Truss/Yen Crisis (Sep 2022)": ("2022-08-01", "2022-11-30"),
        "China Devaluation (Aug 2015)": ("2015-08-10", "2015-09-30"),
    }

    print(f"\n  {'Event':<40} {'CNSI-Z':>8} {'RVol-Z':>8} {'CNSI>2.5':>10} {'RVol>2.5':>10}")
    print(f"  {'-'*80}")
    for ev_name, (start, end) in all_events.items():
        mask = (pd.DatetimeIndex(dates_lag) >= pd.Timestamp(start)) & \
               (pd.DatetimeIndex(dates_lag) <= pd.Timestamp(end))
        if not mask.any():
            print(f"  {ev_name:<40} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
            continue

        # Max z-score in window
        cnsi_in = cnsi_z_series[mask].dropna()
        rvol_in = rvol_z_aligned[mask].dropna()
        cnsi_max = float(cnsi_in.max()) if len(cnsi_in) > 0 else np.nan
        rvol_max = float(rvol_in.max()) if len(rvol_in) > 0 else np.nan

        cnsi_det = "YES" if cnsi_max > 2.5 else "NO "
        rvol_det = "YES" if rvol_max > 2.5 else "NO "
        print(f"  {ev_name:<40} {cnsi_max:>8.2f} {rvol_max:>8.2f} {cnsi_det:>10} {rvol_det:>10}")

    # Dates where CNSI-Z > 2.5 but RVol-Z < 2.5 (CNSI exclusive detections)
    cnsi_only = (cnsi_clean > 2.5) & (rvol_clean < 2.5)
    rvol_only = (rvol_clean > 2.5) & (cnsi_clean < 2.5)
    both_fire = (cnsi_clean > 2.5) & (rvol_clean > 2.5)

    print(f"\n  Threshold crossings (z > 2.5):")
    print(f"    CNSI only (RVol misses): {cnsi_only.sum()} days")
    print(f"    RVol only (CNSI misses): {rvol_only.sum()} days")
    print(f"    Both fire:               {both_fire.sum()} days")
    print(f"    Neither fires:           {(~cnsi_only & ~rvol_only & ~both_fire).sum()} days")

    # False positive analysis: days outside ALL crisis windows that fire
    crisis_mask = np.zeros(len(cnsi_clean), dtype=bool)
    dates_arr = cnsi_clean.index
    for start, end in all_events.values():
        crisis_mask |= (dates_arr >= pd.Timestamp(start)) & (dates_arr <= pd.Timestamp(end))

    non_crisis = ~crisis_mask
    cnsi_fp = (cnsi_clean[non_crisis] > 2.5).sum()
    rvol_fp = (rvol_clean[non_crisis] > 2.5).sum()
    non_crisis_total = non_crisis.sum()

    print(f"\n  False positive rate (non-crisis days):")
    print(f"    CNSI-Z false positives: {cnsi_fp} / {non_crisis_total} non-crisis days ({100*cnsi_fp/non_crisis_total:.1f}%)")
    print(f"    RVol-Z false positives: {rvol_fp} / {non_crisis_total} non-crisis days ({100*rvol_fp/non_crisis_total:.1f}%)")

    return corr_full


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CURRENCY NETWORK — STRUCTURAL ROBUSTNESS ANALYSIS")
    print("=" * 70)

    prices, log_returns, volume_proxy = load_data()
    print(f"\nData: {len(log_returns)} trading days, {len(log_returns.columns)} currencies")

    # Problem 1
    res_lag, cnsi_z_lag, dates_lag = problem1(log_returns, volume_proxy)

    # Problem 2
    all_ranks = problem2(log_returns, volume_proxy, n_bootstrap=500, window=252)

    # Problem 3
    corr_cnsi_rvol = problem3(log_returns, cnsi_z_lag, dates_lag)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
