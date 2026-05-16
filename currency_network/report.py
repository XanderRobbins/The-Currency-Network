"""Report generation: tables, figures, and summary statistics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def save_tables(
    backtest_results,
    calibration_mse,
    beta,
    r2_cn,
    r2_cn_scaled,
    r2_ar1,
    r2_pca,
    gravity_rankings,
    crisis_analysis,
    cn_sharpe,
    naive_sharpe,
    sensitivity_results,
    currencies,
    results_dir="currency_network/results",
):
    """Save all tables as CSV."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # R² comparison (raw and scaled)
    r2_df = pd.DataFrame(
        {
            "Currency": currencies[1:],  # skip USD
            "CN_R2_Raw": [r2_cn.get(c, 0) for c in currencies[1:]],
            "CN_R2_Scaled": [r2_cn_scaled.get(c, 0) for c in currencies[1:]],
            "PCA_R2": [r2_pca.get(c, 0) for c in currencies[1:]],
            "AR1_R2": [r2_ar1.get(c, 0) for c in currencies[1:]],
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

    # Sensitivity analysis
    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_csv(f"{results_dir}/sensitivity_analysis.csv", index=False)

    # CNSI timeseries
    cnsi_df = pd.DataFrame(
        {
            "Date": backtest_results["dates"],
            "Fiedler": backtest_results["fiedler"],
        }
    )
    cnsi_df.to_csv(f"{results_dir}/cnsi_timeseries.csv", index=False)


def plot_cnsi_timeseries(backtest_results, crisis_windows, results_dir="currency_network/results"):
    """Plot CNSI z-score with crisis windows."""
    dates = backtest_results["dates"]
    fiedler = backtest_results["fiedler"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, fiedler, linewidth=2, label="Fiedler Value")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Fiedler Value", fontsize=12)
    ax.set_title("Currency Network: Spectral Gap (Fiedler Value) Over Time", fontsize=14, fontweight="bold")

    # Add crisis windows
    colors = ["red", "orange", "purple"]
    for (crisis_name, (start, end)), color in zip(crisis_windows.items(), colors):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color=color, label=crisis_name)

    ax.legend(loc="best")
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
    currencies_plot = currencies[1:]  # skip USD

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

    currencies_subset = currencies[1:]  # skip USD
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
        cn_r2_scaled = r2_cn_scaled.get(curr, 0)
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
    print("\n--- SENSITIVITY ANALYSIS ---")
    for row in sensitivity_results:
        print(
            f"tau={row['tau']:3d}, {row['method']:<6s}, "
            f"{row['beta_type']:<20s}: avg R² = {row['avg_r2']:.3f}"
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
