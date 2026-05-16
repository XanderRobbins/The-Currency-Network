"""Report generation: tables, figures, and summary statistics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def save_tables(
    backtest_results,
    cnsi_series,
    cnsi_z,
    calibration_mse,
    beta,
    r2_cn,
    r2_cn_scaled,
    r2_ar1,
    r2_pca,
    gravity_rankings,
    crisis_analysis,
    detection_stats,
    cn_sharpe,
    naive_sharpe,
    sensitivity_results,
    currencies,
    results_dir="currency_network/results",
):
    """Save all tables as CSV."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # R² comparison (raw and scaled) — currencies already excludes USD
    r2_df = pd.DataFrame(
        {
            "Currency": currencies,
            "CN_R2_Raw": [r2_cn.get(c, 0) for c in currencies],
            "CN_R2_Scaled": [r2_cn_scaled.get(c, 0) for c in currencies],
            "PCA_R2": [r2_pca.get(c, 0) for c in currencies],
            "AR1_R2": [r2_ar1.get(c, 0) for c in currencies],
        }
    )
    r2_df.to_csv(f"{results_dir}/r2_comparison.csv", index=False)

    # Gravity rankings
    gravity_df = pd.DataFrame(gravity_rankings, columns=["Currency", "Pressure"])
    gravity_df["Rank"] = range(1, len(gravity_df) + 1)
    gravity_df = gravity_df[["Rank", "Currency", "Pressure"]]
    gravity_df.to_csv(f"{results_dir}/gravity_rankings.csv", index=False)

    # Crisis analysis
    crisis_df_list = []
    for crisis_name, stats in crisis_analysis.items():
        crisis_df_list.append(
            {
                "Crisis": crisis_name,
                "Peak_CNSI_Z": stats["peak_zscore"],
                "Peak_Date": stats["peak_date"].strftime("%Y-%m-%d"),
                "Fiedler_Decline_Pct": stats["fiedler_decline"],
                "Weeks_Before": stats["weeks_before"],
                "Lead_Time_Weeks": stats["lead_time"],
            }
        )
    crisis_df = pd.DataFrame(crisis_df_list)
    crisis_df.to_csv(f"{results_dir}/crisis_analysis.csv", index=False)

    # Detection stats
    det_rows = []
    for event, stats in detection_stats["per_crisis"].items():
        det_rows.append({
            "Event": event,
            "Max_CNSI_Z": stats["max_z"],
            "Detected": stats["detected"],
            "Peak_Date": stats["peak_date"].strftime("%Y-%m-%d") if stats["peak_date"] else "",
            "Fiedler_Lead_Weeks": stats["fiedler_lead_weeks"],
        })
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(f"{results_dir}/crisis_detection_stats.csv", index=False)

    # Sensitivity analysis
    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_csv(f"{results_dir}/sensitivity_analysis.csv", index=False)

    # CNSI timeseries
    cnsi_df = pd.DataFrame(
        {
            "Date": backtest_results["dates"],
            "CNSI": cnsi_series,
            "CNSI_Z": cnsi_z,
            "Fiedler": backtest_results["fiedler"],
        }
    )
    cnsi_df.to_csv(f"{results_dir}/cnsi_timeseries.csv", index=False)


def plot_cnsi_timeseries(cnsi_series, cnsi_z, dates, crisis_windows, results_dir="currency_network/results"):
    """Plot CNSI and its z-score with crisis windows."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(dates, cnsi_series, linewidth=2, color="darkblue", label="CNSI")
    ax1.set_ylabel("CNSI (elastic potential energy)", fontsize=12)
    ax1.set_title("Currency Network Stress Index (CNSI)", fontsize=14, fontweight="bold")

    ax2.plot(dates, cnsi_z, linewidth=2, color="darkred", label="CNSI z-score")
    ax2.axhline(2.5, color="orange", linestyle="--", linewidth=1, label="2.5σ threshold")
    ax2.set_ylabel("CNSI z-score (trailing 252-day)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    colors = ["red", "orange", "purple"]
    for ax in [ax1, ax2]:
        for (crisis_name, (start, end)), color in zip(crisis_windows.items(), colors):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color=color, label=crisis_name)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{results_dir}/cnsi_timeseries.png", dpi=150)
    plt.close()


def plot_fiedler_timeseries(backtest_results, crisis_windows, results_dir="currency_network/results"):
    """Plot Fiedler value over time."""
    dates = backtest_results["dates"]
    fiedler = backtest_results["fiedler"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, fiedler, linewidth=2.5, color="darkblue")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Fiedler Value δ(t)", fontsize=12)
    ax.set_title("Spectral Gap: Network Connectivity Over Time", fontsize=14, fontweight="bold")

    colors = ["red", "orange", "purple"]
    for (crisis_name, (start, end)), color in zip(crisis_windows.items(), colors):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color, label=crisis_name)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/fiedler_timeseries.png", dpi=150)
    plt.close()


def plot_r2_comparison(r2_cn, r2_ar1, r2_pca, currencies, results_dir="currency_network/results"):
    """Plot R² comparison across methods."""
    currencies_plot = currencies  # USD already excluded by caller

    cn_vals = [r2_cn.get(c, 0) for c in currencies_plot]
    ar1_vals = [r2_ar1.get(c, 0) for c in currencies_plot]
    pca_vals = [r2_pca.get(c, 0) for c in currencies_plot]

    x = np.arange(len(currencies_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, cn_vals, width, label="CN Model", alpha=0.8)
    ax.bar(x, ar1_vals, width, label="AR(1)", alpha=0.8)
    ax.bar(x + width, pca_vals, width, label="PCA(3)", alpha=0.8)

    ax.set_xlabel("Currency", fontsize=12)
    ax.set_ylabel("R² (Out-of-Sample 2020-2024)", fontsize=12)
    ax.set_title("Displacement Prediction: Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(currencies_plot)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(f"{results_dir}/displacement_r2_bar.png", dpi=150)
    plt.close()


def plot_gravity_ranking(gravity_rankings, results_dir="currency_network/results"):
    """Plot gravitational pressure ranking."""
    currencies = [r[0] for r in gravity_rankings]
    pressures = [r[1] for r in gravity_rankings]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["darkgreen" if p >= 2 * np.mean(pressures) else "steelblue" for p in pressures]
    ax.barh(currencies, pressures, color=colors, alpha=0.8)

    ax.set_xlabel("Gravitational Pressure Score", fontsize=12)
    ax.set_title("Reserve Currency Ranking", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(f"{results_dir}/gravity_ranking.png", dpi=150)
    plt.close()


def plot_mode_decomposition(L_last, currencies, results_dir="currency_network/results"):
    """Plot spectral mode decomposition."""
    evals, evecs = np.linalg.eigh(L_last)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Mode 2 and 3
    mode2 = evecs[:, 1]
    mode3 = evecs[:, 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(currencies, mode2, alpha=0.8, color="steelblue")
    ax1.set_ylabel("Loading", fontsize=11)
    ax1.set_title("Mode 2 (Fiedler Vector)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(currencies, mode3, alpha=0.8, color="coral")
    ax2.set_ylabel("Loading", fontsize=11)
    ax2.set_title("Mode 3", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{results_dir}/mode_decomposition.png", dpi=150)
    plt.close()


def plot_crisis_detection_overview(cnsi_z, fiedler_series, dates, detection_stats, results_dir="currency_network/results"):
    """Three-panel crisis detection overview: CNSI z-score, Fiedler, and per-event bar chart."""
    from currency_network.cnsi import CRISIS_WINDOWS

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [2, 2, 1]})

    # Panel 1: CNSI z-score
    ax1.plot(dates, cnsi_z, linewidth=1.5, color="darkred", label="CNSI z-score")
    ax1.axhline(2.5, color="orange", linestyle="--", linewidth=1.2, label="2.5σ threshold")
    ax1.fill_between(dates, 0, cnsi_z, where=cnsi_z > 2.5, color="red", alpha=0.3, label="Detected stress")
    ax1.set_ylabel("CNSI z-score", fontsize=11)
    ax1.set_title("Crisis Detection: CNSI and Fiedler Early Warning (2015–2024)", fontsize=13, fontweight="bold")

    for (crisis_name, (start, end)), color in zip(CRISIS_WINDOWS.items(), colors):
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color=color)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Fiedler value
    ax2.plot(dates, fiedler_series, linewidth=1.5, color="darkblue", label="Fiedler value mu_2")
    ax2.set_ylabel("Fiedler value δ(t)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)

    for (crisis_name, (start, end)), color in zip(CRISIS_WINDOWS.items(), colors):
        ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color=color, label=crisis_name)
    ax2.legend(loc="upper left", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Per-event max CNSI z-score bar chart
    per_crisis = detection_stats["per_crisis"]
    event_names = [k.split(" (")[0] for k in per_crisis.keys()]
    max_zs = [v["max_z"] if not np.isnan(v["max_z"]) else 0.0 for v in per_crisis.values()]
    bar_colors = ["#d62728" if z > 2.5 else ("#ff7f0e" if z > 1.5 else "#aec7e8") for z in max_zs]

    bars = ax3.barh(event_names, max_zs, color=bar_colors, alpha=0.85)
    ax3.axvline(2.5, color="orange", linestyle="--", linewidth=1.2, label="2.5σ threshold")
    ax3.set_xlabel("Peak CNSI z-score within event window", fontsize=10)
    ax3.set_title(
        f"Per-Event Detection  |  Hit rate: {detection_stats['n_detected']}/{detection_stats['n_valid_windows']} "
        f"({100*detection_stats['detection_rate']:.0f}%)  |  "
        f"False positive rate: {100*detection_stats['fp_rate_daily']:.1f}%/day",
        fontsize=10,
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="x")

    for bar, z in zip(bars, max_zs):
        ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{z:.2f}σ", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{results_dir}/crisis_detection_overview.png", dpi=150)
    plt.close()


def plot_pairs_pnl(cn_pnl, naive_pnl, dates, results_dir="currency_network/results"):
    """Plot cumulative PnL of pairs strategies."""
    cn_cumulative = np.cumsum(cn_pnl)
    naive_cumulative = np.cumsum(naive_pnl)

    # Ensure dates match pnl length
    if len(dates) > len(cn_cumulative):
        dates = dates[:len(cn_cumulative)]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, cn_cumulative, linewidth=2, label="CN Pairs", alpha=0.8)
    ax.plot(dates, naive_cumulative, linewidth=2, label="Naive Pairs", alpha=0.8)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative PnL", fontsize=12)
    ax.set_title("Pairs Trading: EUR/GBP Strategy Performance", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{results_dir}/pairs_pnl.png", dpi=150)
    plt.close()


def print_report(
    beta,
    calibration_mse,
    r2_cn,
    r2_cn_scaled,
    corr_cn,
    pval_cn,
    r2_ar1,
    r2_pca,
    gravity_rankings,
    reserve_currencies,
    crisis_analysis,
    detection_stats,
    cn_sharpe,
    naive_sharpe,
    sensitivity_results,
    displacement_bound_check,
    currencies,
):
    """Print full statistics report."""

    print("\n" + "=" * 60)
    print("  CURRENCY NETWORK — EMPIRICAL RESULTS REPORT")
    print("=" * 60)

    # Mass calibration
    print("\n--- MASS CALIBRATION (in-sample 2015-2019) ---")
    print(f"Calibrated beta: beta1={beta[0]:.3f}, beta2={beta[1]:.3f}, beta3={beta[2]:.3f}")
    print(f"In-sample MSE: {calibration_mse:.6f}")

    # R² comparison (raw)
    print("\n--- DISPLACEMENT PREDICTION R² (raw, out-of-sample 2020-2024, 1-day-ahead) ---")
    print(
        f"{'Currency':<12} | {'CN R²':>7} | {'PCA Recon':>8} | {'AR(1) R²':>8}"
    )
    print("-" * 48)

    currencies_subset = currencies  # USD already excluded by caller
    r2_cn_list = []
    r2_cn_scaled_list = []
    r2_ar1_list = []
    r2_pca_list = []

    for curr in currencies_subset:
        cn_r2 = r2_cn.get(curr, 0)
        ar1_r2 = r2_ar1.get(curr, 0)
        pca_r2 = r2_pca.get(curr, 0)

        r2_cn_list.append(cn_r2)
        r2_ar1_list.append(ar1_r2)
        r2_pca_list.append(pca_r2)

        print(f"{curr:<12} | {cn_r2:7.3f} | {pca_r2:8.3f} | {ar1_r2:8.3f}")

    print("-" * 48)
    print(f"{'G10 Avg':<12} | {np.mean(r2_cn_list):7.3f} | {np.mean(r2_pca_list):8.3f} | {np.mean(r2_ar1_list):8.3f}")

    # R² comparison (scaled)
    print("\n--- DISPLACEMENT PREDICTION R² (scaled by OLS, out-of-sample 2020-2024, 1-day-ahead) ---")
    print(
        f"{'Currency':<12} | {'CN R² (scaled)':>15}"
    )
    print("-" * 30)

    for curr in currencies_subset:
        cn_r2_scaled = r2_cn_scaled.get(curr, 0.0)
        r2_cn_scaled_list.append(cn_r2_scaled)
        print(f"{curr:<12} | {cn_r2_scaled:15.3f}")

    print("-" * 30)
    print(f"{'G10 Avg':<12} | {np.mean(r2_cn_scaled_list):15.3f}")

    # Predictive correlation
    print("\n--- PREDICTIVE CORRELATION: u_star(t) vs u_obs(t+1) ---")
    print(
        f"{'Currency':<12} | {'Pearson r':>11} | {'p-value':>10}"
    )
    print("-" * 36)

    for curr in currencies_subset:
        r = corr_cn.get(curr, 0)
        p = pval_cn.get(curr, 1.0)
        print(f"{curr:<12} | {r:11.4f} | {p:10.4f}")

    print("-" * 36)

    # Crisis analysis
    print("\n--- CNSI CRISIS ANALYSIS ---")
    for crisis_name, stats in crisis_analysis.items():
        print(f"\nCrisis: {crisis_name}")
        print(f"  Peak CNSI z-score: {stats['peak_zscore']:6.2f}s.d.  on {stats['peak_date'].strftime('%Y-%m-%d')}")
        print(
            f"  Fiedler decline before peak: {stats['fiedler_decline']:5.1f}% over {stats['weeks_before']:.0f} weeks"
        )
        print(f"  Lead time (Fiedler to CNSI peak): {stats['lead_time']:.0f} weeks")

    # Crisis detection statistics
    print("\n--- CRISIS DETECTION STATISTICS (threshold = 2.5 s.d.) ---")
    print(
        f"{'Event':<35} | {'Max CNSI-Z':>10} | {'Detected':>8} | {'Fiedler Lead':>12}"
    )
    print("-" * 75)

    per = detection_stats["per_crisis"]
    for event, stats in per.items():
        max_z = stats["max_z"]
        detected = "YES" if stats["detected"] else "no"
        lead = f"{stats['fiedler_lead_weeks']:.1f} wks" if stats["fiedler_lead_weeks"] else "—"
        note = stats.get("note", "")
        if np.isnan(max_z):
            z_str = "  n/a (no history)"
            detected = "—"
        else:
            z_str = f"{max_z:9.2f}sd"
        print(f"{event:<35} | {z_str:>10} | {detected:>8} | {lead:>12}  {note}")

    print("-" * 75)
    print(
        f"Hit rate:            {detection_stats['n_detected']}/{detection_stats['n_valid_windows']} events "
        f"= {100 * detection_stats['detection_rate']:.0f}%"
    )
    print(
        f"False positive rate: {detection_stats['fp_days']} days / {detection_stats['non_crisis_days']} non-crisis days "
        f"= {100 * detection_stats['fp_rate_daily']:.2f}%/day"
    )
    if detection_stats["mean_fiedler_lead_weeks"] > 0:
        print(f"Mean Fiedler lead time (detected events): {detection_stats['mean_fiedler_lead_weeks']:.1f} weeks")

    # Gravity rankings
    print("\n--- GRAVITATIONAL PRESSURE: RESERVE CURRENCY RANKING ---")
    print(f"{'Rank':<5} {'Currency':<12} {'G-Pressure Score':>18}")
    for rank, (curr, pressure) in enumerate(gravity_rankings, 1):
        print(f"{rank:<5} {curr:<12} {pressure:18.4f}")

    print(f"\nReserve currencies (P_grav >= 2x mean): {reserve_currencies}")

    # Pairs trading
    print("\n--- PAIRS TRADING (EUR/GBP, 2020-2024) ---")
    print(f"CN Pairs Sharpe:    {cn_sharpe:7.3f}")
    print(f"Naive Pairs Sharpe: {naive_sharpe:7.3f}")

    # Sensitivity analysis
    print("\n--- SENSITIVITY ANALYSIS (avg scaled R², OOS 2020-2024) ---")
    for row in sensitivity_results:
        print(
            f"tau={row['tau']:3d}, {row['method']:<6s}, "
            f"{row['beta_type']:<20s}: avg scaled R² = {row['avg_r2']:.3f}"
        )

    # Displacement bound check
    print("\n--- DISPLACEMENT BOUND CHECK (Theorem 2 verification) ---")
    print(f"Fraction of dates where bound holds: {displacement_bound_check['fraction']:.1f}%")
    print(f"Mean ratio ||u*||_2 / (||f||_2 / mu_2): {displacement_bound_check['mean_ratio']:.4f}  (should be <= 1.0)")

    # Model specification notes
    print("\n--- MODEL SPECIFICATION NOTES ---")
    print("Volume proxy: BIS Triennial 2022 static weights (USD=88.4%, EUR=31.5%, ...)")
    print("Displacement definition: daily log return minus 60-day rolling mean (stationary)")
    print("Prediction horizon: 1-day-ahead (u*(t) predicts u_obs(t+1))")
    print("PCA benchmark: contemporaneous reconstruction (not prediction)")

    print("\n" + "=" * 60)
    print("  END OF REPORT — all tables saved to results/")
    print("=" * 60 + "\n")
