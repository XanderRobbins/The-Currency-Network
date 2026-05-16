# Currency Network

## Overview

The Currency Network model is a physics-inspired framework for understanding foreign exchange (FX) market dynamics. It represents global currencies as nodes in a spring-mass network, where:

- **Springs** represent correlations between currency pairs
- **Mass** combines volatility, trading volume, and information ratio
- **Network Laplacian** captures the global connectivity structure
- **Equilibrium displacement** predicts currency deviations from their cross-sectional mean
- **CNSI (Currency Network Stress Indicator)** measures systemic FX stress via spectral properties

The model uses historical G10 FX data (EUR, GBP, JPY, CHF, AUD, CAD, NZD, SEK, NOK vs USD) from 2015-2024 to:

1. Predict currency displacements (out-of-sample R²)
2. Detect financial crises via spectral gap (Fiedler value) analysis
3. Rank reserve currencies by gravitational pressure
4. Generate profitable pairs trading signals

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full end-to-end pipeline:

```bash
python main.py
```

This will:
- Download 10 years of daily FX data from Yahoo Finance (cached on first run)
- Calibrate mass parameters on in-sample data (2015-2019)
- Run rolling backtest on out-of-sample period (2020-2024)
- Compute displacement predictions vs. AR(1) and PCA benchmarks
- Analyze three major crises (COVID, Russia/Ukraine, Yen crash)
- Rank currencies by reserve currency pressure (gravitational mechanics)
- Generate pairs trading strategy (EUR/GBP) with Sharpe ratio
- Produce sensitivity analysis across tau, shrinkage, and beta values
- Save all tables as CSV and all figures as PNG to `results/`

## Output

All statistics printed to stdout; results saved to `currency_network/results/`:

### Tables (CSV)
- `r2_comparison.csv` — R² scores per currency (CN vs AR(1) vs PCA)
- `gravity_rankings.csv` — Reserve currency ranking by gravitational pressure
- `crisis_analysis.csv` — Peak CNSI z-scores and Fiedler lead times for 3 crises
- `sensitivity_analysis.csv` — Performance across tau (30, 60, 252), shrinkage, and beta
- `cnsi_timeseries.csv` — CNSI and Fiedler value daily time series

### Figures (PNG)
- `fiedler_timeseries.png` — Spectral gap over time (with crisis windows shaded)
- `cnsi_timeseries.png` — CNSI z-score over time
- `displacement_r2_bar.png` — Model comparison (CN vs AR(1) vs PCA)
- `gravity_ranking.png` — Gravitational pressure bar chart
- `mode_decomposition.png` — Fiedler vector and Mode 3 loadings
- `pairs_pnl.png` — Cumulative PnL of pairs trading (EUR/GBP)

## Data Source

Historical FX data downloaded from Yahoo Finance via `yfinance`:
- **Tickers**: EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, USDCAD=X, NZDUSD=X, USDSEK=X, USDNOK=X
- **Frequency**: Daily adjusted close prices
- **Period**: 2015-01-01 to 2024-12-31 (~2,500 trading days)
- **Inversion**: JPY, CHF, CAD, SEK, NOK stored as 1/cross to align with USD base

## Expected Runtime

- First run: ~5-10 minutes (data download + calibration + backtest)
- Cached runs: ~2-3 minutes (uses local CSV data)

On a modern CPU (4+ cores), the rolling backtest (~1,200 windows) completes in under 2 minutes.

## Reproducibility

- Random seed: `np.random.seed(42)`
- PCA trained on in-sample (2015-2019), applied to out-of-sample (2020-2024)
- AR(1) models fitted per-currency on in-sample, predicted out-of-sample
- Mass parameters calibrated via grid search on in-sample MSE
