"""CNSI (Currency Network Stress Indicator) and crisis analysis."""

import numpy as np
import pandas as pd

CRISIS_WINDOWS = {
    "COVID (Mar 2020)": ("2020-02-01", "2020-04-30"),
    "Russia/Ukraine (Feb 2022)": ("2022-01-15", "2022-04-30"),
    "Yen Crisis (Sep-Oct 2022)": ("2022-08-01", "2022-11-30"),
}


def compute_cnsi(u_star, L):
    """Compute CNSI = 0.5 * u* @ L @ u*."""
    return 0.5 * u_star @ L @ u_star


def compute_cnsi_series(backtest_results):
    """Compute CNSI time series from backtest results."""
    cnsi_list = []

    for i in range(len(backtest_results["u_star"])):
        u_star = backtest_results["u_star"][i]
        L = backtest_results["L_series"][i]
        cnsi = compute_cnsi(u_star, L)
        cnsi_list.append(cnsi)

    return np.array(cnsi_list)


def detect_crisis_spikes(cnsi_series, dates, window=252, threshold=2.5):
    """Detect crisis spikes in CNSI."""
    rolling_mean = pd.Series(cnsi_series).rolling(window=window, center=True).mean().values
    rolling_std = pd.Series(cnsi_series).rolling(window=window, center=True).std().values

    z_score = (cnsi_series - rolling_mean) / (rolling_std + 1e-10)

    crisis_dates = dates[z_score > threshold]

    return z_score, crisis_dates


def analyze_crisis_episodes(cnsi_z, fiedler_series, dates):
    """Analyze crisis episodes."""
    results = {}

    for crisis_name, (start_date, end_date) in CRISIS_WINDOWS.items():
        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))

        if not mask.any():
            continue

        crisis_indices = np.where(mask)[0]

        # Peak CNSI z-score in crisis window
        z_scores_in_window = cnsi_z[crisis_indices]
        valid_mask = ~np.isnan(z_scores_in_window)

        if not valid_mask.any():
            continue

        peak_idx = crisis_indices[np.where(valid_mask)[0][np.argmax(z_scores_in_window[valid_mask])]]
        peak_zscore = float(cnsi_z[peak_idx])
        peak_date = dates[peak_idx]

        # Fiedler 9 weeks before peak vs at peak
        weeks_before = 63  # 9 weeks * 5 trading days
        fiedler_before_idx = max(0, peak_idx - weeks_before)

        if fiedler_before_idx < peak_idx:
            fiedler_before = fiedler_series[fiedler_before_idx : peak_idx].mean()
            fiedler_at_peak = fiedler_series[peak_idx]
            fiedler_decline = (fiedler_before - fiedler_at_peak) / (
                fiedler_before + 1e-10
            ) * 100
            weeks_before_measured = (peak_idx - fiedler_before_idx) / 5
        else:
            fiedler_decline = 0
            weeks_before_measured = 0

        # Lead time: when fiedler drops below 30-day trailing mean by >5%
        window_30d = 30
        lead_time = None

        if peak_idx >= window_30d:
            for check_idx in range(peak_idx - weeks_before, peak_idx):
                if check_idx < window_30d:
                    continue
                mean_before = fiedler_series[check_idx - window_30d : check_idx].mean()
                if fiedler_series[check_idx] < mean_before * 0.95:
                    lead_time = (peak_idx - check_idx) / 5
                    break

        if lead_time is None:
            lead_time = 0

        results[crisis_name] = {
            "peak_zscore": peak_zscore,
            "peak_date": peak_date,
            "fiedler_decline": fiedler_decline,
            "weeks_before": weeks_before_measured,
            "lead_time": lead_time,
        }

    return results
