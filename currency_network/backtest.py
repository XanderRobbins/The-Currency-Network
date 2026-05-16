"""Rolling backtest engine and benchmarks."""

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg

from .model import (
    compute_correlation_matrix,
    compute_spring_constants,
    compute_signed_laplacian,
    compute_mass,
    compute_equilibrium,
    compute_fiedler_value,
    compute_displacement,
)

TAU = 60
START_BACKTEST = "2020-01-01"
END_BACKTEST = "2024-12-31"


def run_rolling_backtest(prices, log_returns, volume_proxy, tau=60, beta=(0.5, 0.3, 0.2)):
    """Run rolling backtest for displacement prediction (1-day-ahead forecasts)."""

    # Compute displacement once for all dates (daily residuals from trend)
    displacement_all = compute_displacement(log_returns, tau)

    # Filter to backtest window using log_returns index (which is aligned)
    mask = (log_returns.index >= START_BACKTEST) & (log_returns.index <= END_BACKTEST)
    dates = log_returns.index[mask]

    results = {
        "dates": [],
        "u_star": [],
        "u_obs": [],
        "fiedler": [],
        "L_series": [],
        "f_series": [],
    }

    for t_idx, date in enumerate(tqdm(dates, desc="Rolling backtest")):
        date_loc = log_returns.index.get_loc(date)

        if date_loc < tau:
            continue

        # Extract window [t-tau, t]
        window_start = date_loc - tau
        window_end = date_loc

        log_ret_window = log_returns.iloc[window_start:window_end].values
        vol_window = volume_proxy.iloc[window_start:window_end].values

        # Skip if too many NaNs
        if np.isnan(log_ret_window).sum() > (tau * log_ret_window.shape[1] / 2):
            continue

        # Compute network quantities at end of day t
        corr = compute_correlation_matrix(log_ret_window, shrinkage=True)
        K, alpha, k0 = compute_spring_constants(corr)
        L = compute_signed_laplacian(K, alpha)

        # Compute mass
        mass = compute_mass(log_ret_window, vol_window, beta=beta)

        # Compute shock vector f(t): use today's displacement (deviation from rolling mean)
        # This is the same quantity as u_obs(t) — the raw shock.
        # u*(t) = L_plus @ f(t) is the network-filtered version of this shock.
        u_obs_date = displacement_all.loc[date].values
        if np.isnan(u_obs_date).any():
            continue

        f = u_obs_date.copy()
        f = f - f.mean()  # enforce sum-to-zero ONCE

        # Compute equilibrium: u*(t) = L_plus @ f(t) is the network-filtered shock
        # u_star(t) predicts u_obs(t+1)
        u_star, L_plus = compute_equilibrium(L, f)

        # Fiedler value
        fiedler = compute_fiedler_value(L)

        results["dates"].append(date)
        results["u_star"].append(u_star)
        results["u_obs"].append(u_obs_date)
        results["fiedler"].append(fiedler)
        results["L_series"].append(L)
        results["f_series"].append(f)

    results["u_star"] = np.array(results["u_star"])
    results["u_obs"] = np.array(results["u_obs"])
    results["fiedler"] = np.array(results["fiedler"])
    results["f_series"] = np.array(results["f_series"])

    return results


def scale_u_star_ols(u_star, u_obs):
    """Scale u_star per currency using OLS regression coefficients.
    For each currency i, fit: u_obs_i ~ scale_i * u_star_i
    Returns scaled u_star."""
    u_star_scaled = u_star.copy()

    for i in range(u_star.shape[1]):
        var_u_star = np.var(u_star[:, i])
        if var_u_star > 1e-10:
            cov_u_star_u_obs = np.cov(u_star[:, i], u_obs[:, i])[0, 1]
            scale_i = cov_u_star_u_obs / var_u_star
        else:
            scale_i = 1.0

        u_star_scaled[:, i] = scale_i * u_star[:, i]

    return u_star_scaled


def compute_r2_per_currency(u_star, u_obs, currencies):
    """Compute 1-day-ahead prediction R² for each currency.
    u_star[t] predicts u_obs[t+1], so shift and align."""
    r2_dict = {}
    r2_scaled_dict = {}

    # Forward-looking: u_star[t] predicts u_obs[t+1]
    u_star_shifted = u_star[:-1]   # predictions at t
    u_obs_target = u_obs[1:]        # realized at t+1

    # Scale u_star
    u_star_scaled = scale_u_star_ols(u_star_shifted, u_obs_target)

    for i, curr in enumerate(currencies):
        # Raw R²
        ss_res = np.sum((u_obs_target[:, i] - u_star_shifted[:, i]) ** 2)
        ss_tot = np.sum((u_obs_target[:, i] - u_obs_target[:, i].mean()) ** 2)

        if ss_tot > 1e-10:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = 0.0

        r2_dict[curr] = max(r2, 0.0)  # floor at 0

        # Scaled R²
        ss_res_scaled = np.sum((u_obs_target[:, i] - u_star_scaled[:, i]) ** 2)
        ss_tot_scaled = np.sum((u_obs_target[:, i] - u_obs_target[:, i].mean()) ** 2)

        if ss_tot_scaled > 1e-10:
            r2_scaled = 1.0 - ss_res_scaled / ss_tot_scaled
        else:
            r2_scaled = 0.0

        r2_scaled_dict[curr] = max(r2_scaled, 0.0)

    return r2_dict, r2_scaled_dict


def compute_predictive_correlation(u_star, u_obs, currencies):
    """Compute Pearson correlation and p-value between u_star(t) and u_obs(t+1)."""
    from scipy import stats

    corr_dict = {}
    pval_dict = {}

    # Forward-looking: u_star[t] predicts u_obs[t+1]
    u_star_shifted = u_star[:-1]   # predictions at t
    u_obs_target = u_obs[1:]        # realized at t+1

    for i, curr in enumerate(currencies):
        r, p = stats.pearsonr(u_star_shifted[:, i], u_obs_target[:, i])
        corr_dict[curr] = r
        pval_dict[curr] = p

    return corr_dict, pval_dict


def ar1_benchmark(u_obs, currencies):
    """Fit AR(1) benchmark."""
    r2_dict = {}

    for i, curr in enumerate(currencies):
        u_i = u_obs[:, i]

        try:
            # Fit AR(1)
            model = AutoReg(u_i, lags=1)
            result = model.fit()
            u_pred = result.predict(start=1, end=len(u_i) - 1)

            ss_res = np.sum((u_i[1:] - u_pred) ** 2)
            ss_tot = np.sum((u_i[1:] - u_i[1:].mean()) ** 2)

            if ss_tot > 1e-10:
                r2 = 1.0 - ss_res / ss_tot
            else:
                r2 = 0.0

            r2_dict[curr] = max(r2, 0.0)
        except:
            r2_dict[curr] = 0.0

    return r2_dict


def pca_benchmark(u_obs_train, u_obs_test, currencies, n_components=3):
    """Fit PCA on training set, evaluate on test set."""
    from sklearn.decomposition import PCA

    r2_dict = {}

    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    pca.fit(u_obs_train)

    # Project test data
    u_test_transformed = pca.transform(u_obs_test)
    u_test_reconstructed = pca.inverse_transform(u_test_transformed)

    for i, curr in enumerate(currencies):
        ss_res = np.sum((u_obs_test[:, i] - u_test_reconstructed[:, i]) ** 2)
        ss_tot = np.sum((u_obs_test[:, i] - u_obs_test[:, i].mean()) ** 2)

        if ss_tot > 1e-10:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = 0.0

        r2_dict[curr] = max(r2, 0.0)

    return r2_dict


def calibrate_beta(prices, log_returns, volume_proxy, tau=60):
    """Calibrate beta on in-sample data (2015-2019)."""

    in_sample_mask_returns = log_returns.index < "2020-01-01"
    log_returns_is = log_returns[in_sample_mask_returns]
    volume_proxy_is = volume_proxy[in_sample_mask_returns]

    # Also filter prices to same index
    prices_is = prices.loc[log_returns_is.index]

    displacement_is = compute_displacement(log_returns_is, tau)

    best_mse = np.inf
    best_beta = (0.333, 0.333, 0.334)
    grid_results = []

    # Search beta simplex with step 0.1
    for b1 in np.arange(0, 1.1, 0.1):
        for b2 in np.arange(0, 1.1 - b1, 0.1):
            b3 = 1.0 - b1 - b2
            if b3 < 0 or b3 > 1.0 or b1 < 0.01 or b2 < 0.01 or b3 < 0.01:
                continue

            beta = (b1, b2, b3)

            # Run rolling backtest on in-sample
            dates_is = prices_is.index[prices_is.index >= (prices_is.index[0] + pd.Timedelta(days=tau))]

            mse_total = 0
            count = 0

            for date in dates_is:
                date_loc = prices_is.index.get_loc(date)

                if date_loc < tau:
                    continue

                window_start = date_loc - tau
                window_end = date_loc

                log_ret_window = log_returns_is.iloc[window_start:window_end].values
                vol_window = volume_proxy_is.iloc[window_start:window_end].values

                if np.isnan(log_ret_window).sum() > (tau * log_ret_window.shape[1] / 2):
                    continue

                corr = compute_correlation_matrix(log_ret_window, shrinkage=True)
                K, _, _ = compute_spring_constants(corr)
                L = compute_signed_laplacian(K, np.sign(corr))

                mass = compute_mass(log_ret_window, vol_window, beta=beta)

                current_returns = log_returns_is.iloc[window_end].values
                f = current_returns - current_returns.mean()
                f = f - f.mean()

                u_star, _ = compute_equilibrium(L, f)

                u_obs = displacement_is.loc[date].values
                if np.isnan(u_obs).any():
                    continue

                mse = np.mean((u_star - u_obs) ** 2)
                mse_total += mse
                count += 1

            if count > 0:
                avg_mse = mse_total / count
                grid_results.append((beta, avg_mse))

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_beta = beta

    # Sort and print top 5
    grid_results.sort(key=lambda x: x[1])
    print("\nTop 5 beta combinations (in-sample 2015-2019):")
    for i, (beta, mse) in enumerate(grid_results[:5]):
        print(f"  {i+1}. beta={beta}, MSE={mse:.8f}")

    print(f"\nCalibrated beta: {best_beta}, MSE={best_mse:.8f}")

    return best_beta, best_mse
