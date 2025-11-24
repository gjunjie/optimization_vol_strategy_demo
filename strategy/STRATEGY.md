# Volatility Arbitrage Strategy

## Overview

A statistical arbitrage strategy that identifies mispriced options by detecting deviations from the implied volatility surface.

**Core Concept**: When an option's implied volatility (IV) deviates significantly from the fitted volatility smile:
- **IV below expected** → **Buy Signal** (option is undervalued)
- **IV above expected** → **Sell Signal** (option is overvalued)

## How It Works

### 1. Rolling Window Framework
- **Training Window**: 10 minutes to fit volatility surface
- **Testing Window**: 1 minute to generate trading signals
- Window slides forward continuously through the trading day

### 2. Volatility Surface Model
- **Moneyness Calculation**: `Strike / Forward Price`
- **Polynomial Fitting**: Cubic polynomial `IV = f(Moneyness)`
- The fitted model represents "fair" IV based on recent market data

### 3. Signal Generation

```
Residual = Actual IV - Predicted IV
Z-Score = (Residual - Mean) / StdDev

Buy:  Z-Score ≤ -2.0  (undervalued)
Sell: Z-Score ≥ +2.0  (overvalued)
```

**Filters Applied**:
- Only trade options with `0.3 ≤ Moneyness ≤ 1.7` (avoids deep OTM/ITM options)
- Minimum data quality thresholds for reliable IV calculation

## Current Results

**TSLA 2020-03-20 Test Run**:
- 378 one-minute windows analyzed
- 28 Buy signals generated (7.4%)
- 85 Sell signals generated (22.5%)
- Approximately 30% of windows produce actionable signals

## Key Risks

1. **Model Risk**: Cubic polynomial may not fully capture the true IV surface dynamics
2. **Execution Risk**: Bid-ask spreads, slippage, and latency can erode theoretical profits
3. **Market Risk**: Volatility clustering, jump risk, and directional exposure require hedging
4. **Data Quality**: Missing or stale quotes can generate false signals

## Professional-Grade Improvements

### 1. Advanced Smile Modeling
- **Local Regression (LOESS)**: better handling of local curvature and outliers compared to global polynomials.
- **Arbitrage-Free Surfaces**: Implement SVI or SABR models to enforce no-arbitrage conditions (monotonicity, convexity).
- **Log-Moneyness**: Fit IV against log-moneyness for smoother short-dated curves.

### 2. Signal Quality Enhancement
- **Vega-Normalized Residuals**: Scale residuals by Vega to make IV deviations comparable across strikes.
- **Neighbor-Based Residuals**: Use local butterfly relationships (`IV_cheap - (IV_left + IV_right)/2`) for pure relative value signals.
- **Predictive Residuals**: Forecast the *next* residual using features like volume imbalance and order flow, rather than just mean-reversion.

### 3. Trading Logic & Execution
- **Liquidity Filters**: Enforce strict checks on bid/ask width, quote age, and frequency before entering.
- **Vega-Neutral Structure**: Execute "Fly RV" trades (Long cheap / Short neighbors) to isolate vol mispricing and minimize directional risk.
- **Regime-Aware Sizing**: Adjust position sizes based on market regimes (Calm vs. Chaotic) using volatility-of-volatility and skew metrics.

### 4. System Stability
- **Mean-Reversion Speed**: Model the expected half-life of residuals.
- **Stability Checks**: Only trade when the entire volatility surface is stable; avoid trading during macro events or surface shifts.
- **Transaction Cost Modeling**: Incorporate realistic slippage and spread models to ensure `Residual > 1.5 * Cost`.

## So Many Things to Explore Next
