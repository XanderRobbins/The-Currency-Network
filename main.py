"""Main entry point: run full Currency Network pipeline."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed
np.random.seed(42)

from currency_network.data import download_data, compute_log_returns, compute_volume_proxy, CURRENCIES
from currency_network.backtest import (
    run_rolling_backtest,
    compute_r2_per_currency,
    compute_predictive_correlation,
    ar1_benchmark,
    pca_benchmark,
    calibrate_beta,
    TAU,
    START_BACKTEST,
)
from currency_network.model import compute_displacement
from currency_network.cnsi import compute_cnsi_series, analyze_crisis_episodes, detect_crisis_spikes, CRISIS_WINDOWS
from currency_network.gravity import compute_gravitational_pressure, rank_reserve_currencies
from currency_network.pairs import backtest_pairs_sharpe
from currency_network.report import (
    save_tables,
    plot_cnsi_timeseries,
    plot_fiedler_timeseries,
    plot_r2_comparison,
    plot_gravity_ranking,
    plot_mode_decomposition,
    plot_pairs_pnl,
    print_report,
)


def run_sensitivity_analysis(prices, log_returns, volume_proxy, calibrated_beta):
    """Run sensitivity analysis on tau, shrinkage, and beta."""
    results = []

    for tau in [30, 60, 252]:
        for shrinkage in [True, False]:
            # Run backtest with calibrated beta
            backtest_res = run_rolling_backtest(prices, log_returns, volume_proxy, tau=tau, beta=calibrated_beta)

            if len(backtest_res["u_star"]) > 0:
                r2_dict, r2_scaled_dict = compute_r2_per_currency(backtest_res["u_star"], backtest_res["u_obs"], CURRENCIES[1:])
                avg_r2 = np.mean(list(r2_dict.values()))

                method = "LW" if shrinkage else "sample"
                baseline = "[BASELINE]" if tau == 60 and shrinkage else ""

                results.append(
                    {
                        "tau": tau,
                        "method": method,
                        "beta_type": "calibrated",
                        "avg_r2": avg_r2,
                        "baseline": baseline,
                    }
                )

    # Alternative beta combinations
    for beta_tuple in [(0.333, 0.333, 0.334), (0.70, 0.20, 0.10)]:
        backtest_res = run_rolling_backtest(prices, log_returns, volume_proxy, tau=60, beta=beta_tuple)

        if len(backtest_res["u_star"]) > 0:
            r2_dict, r2_scaled_dict = compute_r2_per_currency(backtest_res["u_star"], backtest_res["u_obs"], CURRENCIES[1:])
            avg_r2 = np.mean(list(r2_dict.values()))

            beta_name = "equal beta" if beta_tuple == (0.333, 0.333, 0.334) else "volume-only beta"

            results.append(
                {
                    "tau": 60,
                    "method": "LW",
                    "beta_type": beta_name,
                    "avg_r2": avg_r2,
                    "baseline": "",
                }
            )

    return results


def compute_displacement_bound_check(backtest_results):
    """Check displacement bound ||u*||_2 <= ||f||_2 / mu_2.

    For each date t:
        numerator   = np.linalg.norm(u_star[t])
        denominator = np.linalg.norm(f[t]) / fiedler[t]
        ratio[t]    = numerator / denominator

    Bound holds if ratio <= 1.0
    """
    u_star = backtest_results["u_star"]
    f_series = backtest_results["f_series"]
    fiedler = backtest_results["fiedler"]

    ratio_list = []
    bound_holds_count = 0

    for i in range(len(u_star)):
        u_norm = np.linalg.norm(u_star[i])
        f_norm = np.linalg.norm(f_series[i])

        if fiedler[i] > 1e-10 and f_norm > 1e-10:
            denominator = f_norm / fiedler[i]
            ratio = u_norm / denominator
            ratio_list.append(ratio)

            if ratio <= 1.0:
                bound_holds_count += 1

    fraction = 100.0 * bound_holds_count / max(len(u_star), 1)
    mean_ratio = np.mean(ratio_list) if ratio_list else 0

    return {"fraction": fraction, "mean_ratio": mean_ratio}


def main():
    """Run full pipeline."""
    print("\n" + "=" * 60)
    print("  CURRENCY NETWORK — FULL PROJECT BUILD")
    print("=" * 60)

    # 1. Download data
    print("\n[1/8] Downloading FX data...")
    prices = download_data()
    print(f"     Loaded {len(prices)} trading days for {len(CURRENCIES)} currencies")

    # 2. Compute returns and volume
    print("\n[2/8] Computing log returns and volume proxy...")
    log_returns = compute_log_returns(prices)
    volume_proxy = compute_volume_proxy(log_returns)
    print(f"     Log returns shape: {log_returns.shape}")

    # 3. Calibrate beta on in-sample
    print("\n[3/8] Calibrating mass parameters (beta)...")
    calibrated_beta, calibration_mse = calibrate_beta(prices, log_returns, volume_proxy, tau=TAU)

    # 4. Run rolling backtest
    print("\n[4/8] Running rolling backtest (out-of-sample 2020-2024)...")
    backtest_results = run_rolling_backtest(prices, log_returns, volume_proxy, tau=TAU, beta=calibrated_beta)
    print(f"     Completed {len(backtest_results['u_star'])} rolling windows")

    # 5. Compute benchmarks
    print("\n[5/8] Running AR(1) and PCA benchmarks...")

    # Get in-sample u_obs for PCA training
    displacement_all = compute_displacement(log_returns, TAU)
    mask_is = displacement_all.index < "2020-01-01"
    mask_oos = (displacement_all.index >= START_BACKTEST) & (displacement_all.index <= "2024-12-31")

    u_obs_is = displacement_all[mask_is].dropna().values
    u_obs_oos = displacement_all[mask_oos].dropna().values[: len(backtest_results["u_star"])]

    # Diagnostic output: check that u_star and f are nonzero
    print("\n     --- DIAGNOSTICS ---")
    print(f"     u_star sample (first 5 dates, all currencies):")
    print(f"     {backtest_results['u_star'][:5]}")
    print(f"     u_star std per currency: {backtest_results['u_star'].std(axis=0)}")
    print(f"     u_obs std per currency: {backtest_results['u_obs'].std(axis=0)}")
    print(f"     f_series sample (first 5 dates):")
    print(f"     {backtest_results['f_series'][:5]}")
    print(f"     f_series std per currency: {backtest_results['f_series'].std(axis=0)}")
    print()

    # Compute R² (both raw and scaled) and correlation
    r2_cn, r2_cn_scaled = compute_r2_per_currency(backtest_results["u_star"], backtest_results["u_obs"], CURRENCIES)
    corr_cn, pval_cn = compute_predictive_correlation(backtest_results["u_star"], backtest_results["u_obs"], CURRENCIES)
    r2_ar1 = ar1_benchmark(backtest_results["u_obs"], CURRENCIES)
    r2_pca = pca_benchmark(u_obs_is, backtest_results["u_obs"], CURRENCIES, n_components=3)

    print(f"     CN avg R² (raw): {np.mean(list(r2_cn.values())):.3f}")
    print(f"     CN avg R² (scaled): {np.mean(list(r2_cn_scaled.values())):.3f}")
    print(f"     AR(1) avg R²: {np.mean(list(r2_ar1.values())):.3f}")
    print(f"     PCA avg R²: {np.mean(list(r2_pca.values())):.3f}")

    # 6. Gravity and reserve currency ranking
    print("\n[6/8] Computing gravitational pressure...")

    # Use last available Laplacian for gravity
    L_last = backtest_results["L_series"][-1]
    K_last = np.abs(L_last - np.diag(np.diag(L_last)))
    K_last = np.diag(np.diag(L_last)) - L_last  # recover K from L

    # Estimate mass from last window
    window_idx = len(log_returns) - TAU
    log_ret_window = log_returns.iloc[-TAU:].values
    vol_window = volume_proxy.iloc[-TAU:].values
    from currency_network.model import compute_mass

    mass_last = compute_mass(log_ret_window, vol_window, beta=calibrated_beta)

    P_grav = compute_gravitational_pressure(K_last, mass_last, G0_normalize=True)
    gravity_rankings, reserve_currencies = rank_reserve_currencies(P_grav, CURRENCIES, theta=2.0)

    # 7. CNSI and crisis analysis
    print("\n[7/8] Analyzing CNSI and crisis episodes...")

    cnsi_series = compute_cnsi_series(backtest_results)
    cnsi_z, crisis_dates = detect_crisis_spikes(cnsi_series, np.array(backtest_results["dates"]), window=252)

    crisis_analysis = analyze_crisis_episodes(
        cnsi_z, backtest_results["fiedler"], np.array(backtest_results["dates"])
    )

    # 8. Pairs trading
    print("\n[8/8] Backtesting pairs trading strategy...")

    cn_sharpe, naive_sharpe, cn_pnl, naive_pnl = backtest_pairs_sharpe(
        backtest_results["u_obs"],
        backtest_results["u_star"],
        log_returns.iloc[: len(backtest_results["u_obs"])],
        CURRENCIES,
        pair=("EUR", "GBP"),
    )

    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity_results = run_sensitivity_analysis(prices, log_returns, volume_proxy, calibrated_beta)

    # Displacement bound check
    displacement_bound = compute_displacement_bound_check(backtest_results)

    # Generate report
    print("\nGenerating report and figures...")

    save_tables(
        backtest_results,
        calibration_mse,
        calibrated_beta,
        r2_cn,
        r2_cn_scaled,
        r2_ar1,
        r2_pca,
        gravity_rankings,
        crisis_analysis,
        cn_sharpe,
        naive_sharpe,
        sensitivity_results,
        CURRENCIES,
    )

    plot_fiedler_timeseries(backtest_results, CRISIS_WINDOWS)
    plot_cnsi_timeseries(backtest_results, CRISIS_WINDOWS)
    plot_r2_comparison(r2_cn, r2_ar1, r2_pca, CURRENCIES)
    plot_gravity_ranking(gravity_rankings)
    plot_mode_decomposition(backtest_results["L_series"][-1], CURRENCIES)
    plot_pairs_pnl(cn_pnl, naive_pnl, np.array(backtest_results["dates"]))

    # Print full report
    print_report(
        calibrated_beta,
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
        displacement_bound,
        CURRENCIES,
    )

    print("\n[OK] All results saved to currency_network/results/")


if __name__ == "__main__":
    main()
