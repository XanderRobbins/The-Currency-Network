"""CNSI (Currency Network Stress Indicator) and crisis analysis."""

import numpy as np
import pandas as pd

# G10 FX stress events 2015-2024.
# CHF Depeg (Jan 15, 2015) is excluded — it occurs before the CNSI z-score has enough
# history (min_periods=126 days), so detection is not meaningful.
CRISIS_WINDOWS = {
    "China Devaluation (Aug 2015)": ("2015-08-10", "2015-09-30"),
    "Brexit (Jun 2016)": ("2016-06-20", "2016-07-31"),
    "US Election (Nov 2016)": ("2016-11-07", "2016-11-30"),
    "COVID (Mar 2020)": ("2020-02-01", "2020-04-30"),
    "Russia/Ukraine (Feb 2022)": ("2022-01-15", "2022-04-30"),
    "Truss/Yen Crisis (Sep 2022)": ("2022-08-01", "2022-11-30"),
    "SVB Crisis (Mar 2023)": ("2023-03-01", "2023-04-30"),
    "Japan Carry Unwind (Aug 2024)": ("2024-07-25", "2024-09-15"),
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
    """Detect crisis spikes in CNSI using trailing rolling z-score."""
    rolling_mean = pd.Series(cnsi_series).rolling(window=window, min_periods=window // 2).mean().values
    rolling_std = pd.Series(cnsi_series).rolling(window=window, min_periods=window // 2).std().values

    z_score = (cnsi_series - rolling_mean) / (rolling_std + 1e-10)

    crisis_dates = dates[z_score > threshold]

    return z_score, crisis_dates


def compute_detection_stats(cnsi_z, fiedler_series, dates, threshold=2.5):
    """Compute comprehensive crisis detection statistics.

    For each known crisis window:
      - Max CNSI z-score within the window
      - Whether the threshold was crossed (detected)
      - Fiedler lead time: weeks before the CNSI peak that Fiedler started declining

    Across all windows:
      - Detection rate (hit rate)
      - False positive rate on non-crisis days
      - Days above threshold outside any crisis window
    """
    # Build union crisis mask
    crisis_mask = np.zeros(len(dates), dtype=bool)
    for start, end in CRISIS_WINDOWS.values():
        crisis_mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    valid_mask = ~np.isnan(cnsi_z)
    detected_mask = (cnsi_z > threshold) & valid_mask

    per_crisis = {}
    for crisis_name, (start, end) in CRISIS_WINDOWS.items():
        window_mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        in_window = window_mask & valid_mask

        if not in_window.any():
            per_crisis[crisis_name] = {
                "max_z": np.nan,
                "detected": False,
                "peak_date": None,
                "fiedler_lead_weeks": None,
                "note": "outside z-score history",
            }
            continue

        indices = np.where(in_window)[0]
        peak_local = np.argmax(cnsi_z[indices])
        peak_idx = indices[peak_local]
        max_z = float(cnsi_z[peak_idx])
        peak_date = dates[peak_idx]

        # Fiedler lead time: how many weeks before peak_idx did Fiedler start declining
        # (drops below its own 30-day trailing mean by >5%)
        lead_weeks = _fiedler_lead_time(fiedler_series, peak_idx, lookback=63, mean_window=30)

        per_crisis[crisis_name] = {
            "max_z": max_z,
            "detected": max_z > threshold,
            "peak_date": peak_date,
            "fiedler_lead_weeks": lead_weeks,
        }

    # Global stats
    fp_days = detected_mask & ~crisis_mask
    non_crisis_valid = valid_mask & ~crisis_mask

    n_detected = sum(1 for v in per_crisis.values() if v["detected"])
    n_windows = len(CRISIS_WINDOWS)
    # Windows that had valid z-score data (exclude "outside z-score history")
    n_valid_windows = sum(1 for v in per_crisis.values() if not np.isnan(v["max_z"]))

    fp_rate = fp_days.sum() / max(non_crisis_valid.sum(), 1)

    # Lead time stats across detected crises
    lead_times = [v["fiedler_lead_weeks"] for v in per_crisis.values()
                  if v["detected"] and v["fiedler_lead_weeks"] is not None]

    return {
        "per_crisis": per_crisis,
        "n_detected": n_detected,
        "n_valid_windows": n_valid_windows,
        "n_total_windows": n_windows,
        "detection_rate": n_detected / max(n_valid_windows, 1),
        "fp_days": int(fp_days.sum()),
        "non_crisis_days": int(non_crisis_valid.sum()),
        "fp_rate_daily": float(fp_rate),
        "mean_fiedler_lead_weeks": float(np.mean(lead_times)) if lead_times else 0.0,
        "threshold": threshold,
    }


def _fiedler_lead_time(fiedler_series, peak_idx, lookback=63, mean_window=30):
    """Return weeks before peak_idx at which Fiedler first dropped >5% below its trailing mean."""
    search_start = max(mean_window, peak_idx - lookback)
    for check_idx in range(search_start, peak_idx):
        trailing_mean = fiedler_series[check_idx - mean_window:check_idx].mean()
        if trailing_mean > 1e-10 and fiedler_series[check_idx] < trailing_mean * 0.95:
            return (peak_idx - check_idx) / 5.0
    return 0.0


def analyze_crisis_episodes(cnsi_z, fiedler_series, dates):
    """Analyze crisis episodes (kept for backward compatibility with main.py)."""
    results = {}

    for crisis_name, (start_date, end_date) in CRISIS_WINDOWS.items():
        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))

        if not mask.any():
            continue

        crisis_indices = np.where(mask)[0]
        z_scores_in_window = cnsi_z[crisis_indices]
        valid_mask = ~np.isnan(z_scores_in_window)

        if not valid_mask.any():
            continue

        peak_idx = crisis_indices[np.where(valid_mask)[0][np.argmax(z_scores_in_window[valid_mask])]]
        peak_zscore = float(cnsi_z[peak_idx])
        peak_date = dates[peak_idx]

        weeks_before = 63
        fiedler_before_idx = max(0, peak_idx - weeks_before)

        if fiedler_before_idx < peak_idx:
            fiedler_before = fiedler_series[fiedler_before_idx:peak_idx].mean()
            fiedler_at_peak = fiedler_series[peak_idx]
            fiedler_decline = (fiedler_before - fiedler_at_peak) / (fiedler_before + 1e-10) * 100
            weeks_before_measured = (peak_idx - fiedler_before_idx) / 5
        else:
            fiedler_decline = 0
            weeks_before_measured = 0

        lead_time = _fiedler_lead_time(fiedler_series, peak_idx)

        results[crisis_name] = {
            "peak_zscore": peak_zscore,
            "peak_date": peak_date,
            "fiedler_decline": fiedler_decline,
            "weeks_before": weeks_before_measured,
            "lead_time": lead_time,
        }

    return results
