# Options Pricing & Volatility Arbitrage

A high-performance quantitative finance toolkit for options pricing, volatility analysis, and statistical arbitrage strategies. Features optimized Black-76 model implementations achieving 1,800x speedup and rolling-window volatility arbitrage strategies for real-time trading signal generation.

## Project Structure

- **benchmark/**: Benchmarking models and performance analysis.
- **data/**: Data files and selection documentation.
- **optimization/**: Optimization models and algorithms (e.g., Black76).
- **query/**: Data querying and loading utilities.
- **strategy/**: Trading strategies, including volatility arbitrage and rolling strategies.
- **utils/**: Utility functions (e.g., interest rates).

## Data Description

### Data Source

The dataset consists of **Algoseek US Options Trades minute bar data** - a free sample providing options market data for demonstration and analysis purposes. The data focuses on equity options from high-volume, highly volatile stocks to ensure sufficient liquidity and interesting pricing dynamics.

### Available Data

The dataset includes minute-by-minute options trading data for the following stocks:

- **AMZN (Amazon)**: 4 expiry days
  - 2020-01-31, 2020-03-20, 2020-06-19, 2020-09-18

- **GOOG (Google)**: 7 expiry days
  - 2020-01-31, 2020-02-14, 2020-03-20, 2020-04-17, 2020-06-19, 2020-09-18, 2021-01-15

- **TSLA (Tesla)**: 5 expiry days
  - 2020-01-31, 2020-02-14, 2020-03-20, 2020-06-19, 2020-09-18

### Data Format

Each CSV file contains minute-bar aggregated options trade data with the following columns:

- **Date**: Trading date (YYYYMMDD format)
- **TimeBarStart**: Minute bar start time (HH:MM format, e.g., "09:30")
- **Ticker**: Underlying stock symbol
- **CallPut**: Option type ("C" for Call, "P" for Put)
- **Strike**: Strike price of the option
- **ExpirationDate**: Option expiration date (YYYYMMDD format)
- **OpenTradePrice**: Opening trade price for the minute
- **HighTradePrice**: Highest trade price for the minute
- **LowTradePrice**: Lowest trade price for the minute
- **CloseTradePrice**: Closing trade price for the minute
- **UnderOpenBidPrice**: Underlying stock opening bid price
- **UnderOpenAskPrice**: Underlying stock opening ask price
- **UnderCloseBidPrice**: Underlying stock closing bid price
- **UnderCloseAskPrice**: Underlying stock closing ask price
- **VWAP**: Volume-weighted average price
- **Volume**: Number of contracts traded
- **TotalTrades**: Total number of trades in the minute

```

For more details on data selection criteria, see [data/DATA_SELECTION.md](data/DATA_SELECTION.md).

## Performance Improvements

The optimization work in this repository demonstrates significant performance gains in options pricing calculations:

### Black-76 Model Optimization

**Results:**
- **Original Benchmark**: ~45 seconds (row-by-row processing)
- **Optimized Model**: ~0.0136 seconds (fully vectorized)
- **Speedup**: Approximately **1,800x faster**

### Key Optimization Techniques

1. **Vectorization**: Replaced pandas `.iterrows()` loops with NumPy array operations
2. **Pre-allocated Arrays**: Eliminated dynamic list appending overhead
3. **Batch Processing**: Separate vectorized processing for calls and puts
4. **Cached Interpolation**: Pre-calculated risk-free rates for unique expiry values
5. **Efficient Data Types**: Fast character comparison for option type detection
6. **Improved Root-Finding**: Newton-Raphson method (3-5 iterations) vs. Bisection (20-30+ iterations)
7. **Optimized CDF/PDF**: Replaced `scipy.stats.norm.cdf()` with `scipy.special.ndtr()`
8. **Adaptive Initial Guess**: Better starting points using Brenner-Subrahmanyam and Corrado-Miller approximations

The optimized model processes 30 minutes of trading data in a fraction of a second, making it suitable for real-time or high-frequency options pricing applications.

For more details, see [optimization/OPTIMIZATION.md](optimization/OPTIMIZATION.md) and [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md).

## Volatility Arbitrage Strategy

A statistical arbitrage strategy that identifies mispriced options by detecting deviations from the implied volatility surface.

### Core Concept

The strategy identifies when an option's implied volatility (IV) deviates significantly from the fitted volatility smile:
- **IV below expected** → **Buy Signal** (option is undervalued)
- **IV above expected** → **Sell Signal** (option is overvalued)

### How It Works

1. **Rolling Window Framework**
   - Training Window: 10 minutes to fit volatility surface
   - Testing Window: 1 minute to generate trading signals
   - Window slides forward continuously through the trading day

2. **Volatility Surface Model**
   - Moneyness Calculation: `Strike / Forward Price`
   - Polynomial Fitting: Cubic polynomial `IV = f(Moneyness)`
   - The fitted model represents "fair" IV based on recent market data

3. **Signal Generation**
   ```
   Residual = Actual IV - Predicted IV
   Z-Score = (Residual - Mean) / StdDev
   
   Buy:  Z-Score ≤ -2.0  (undervalued)
   Sell: Z-Score ≥ +2.0  (overvalued)
   ```

   **Filters Applied:**
   - Only trade options with `0.3 ≤ Moneyness ≤ 1.7` (avoids deep OTM/ITM options)
   - Minimum data quality thresholds for reliable IV calculation

### Performance Results

**TSLA 2020-03-20 Test Run:**
- 378 one-minute windows analyzed
- 28 Buy signals generated (7.4%)
- 85 Sell signals generated (22.5%)
- Approximately 30% of windows produce actionable signals

### Key Features

- Real-time volatility surface fitting using rolling windows
- Statistical signal generation based on Z-scores
- Moneyness-based filtering to avoid illiquid options
- Comprehensive visualization tools for strategy analysis

For more details, see [strategy/STRATEGY.md](strategy/STRATEGY.md).

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

