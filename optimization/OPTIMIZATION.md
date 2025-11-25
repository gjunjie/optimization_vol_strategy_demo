# Optimization Method

## Overview

The optimization transforms the Black-76 options pricing model from a row-by-row processing approach to a fully vectorized implementation using NumPy operations.

## Optimization Techniques

### 1. **Vectorization**
   - Replaced pandas `.iterrows()` loop with NumPy array operations
   - All calculations are performed on entire arrays at once instead of row-by-row

### 2. **Pre-allocated Arrays**
   - Pre-allocate NumPy arrays for all output columns (ImpliedVol, Delta, Gamma, Vega, Theta, Rho)
   - Eliminates dynamic list appending overhead

### 3. **Batch Processing**
   - Separate calls and puts using boolean masks
   - Process each option type in a single vectorized batch

### 4. **Cached Interpolation**
   - Pre-calculate risk-free rates for unique days-to-expiry values
   - Cache results in a dictionary to avoid redundant calculations

### 5. **Efficient Data Type Handling**
   - Convert to NumPy arrays once at the beginning
   - Use fast character comparison for option type detection (no regex or string operations)

### 6. **Improved Initial Guess**
   - Uses adaptive volatility approximations (Brenner-Subrahmanyam for ATM, Corrado-Miller for OTM/ITM)
   - Provides better starting points, reducing iterations needed for convergence

### 7. **Root-Finding Method**
   - Replaced Bisection method with Newton-Raphson method (quadratic convergence)
   - Vectorized implementation with early convergence detection
   - Typically converges in 3-5 iterations vs 20-30+ for bisection

### 8. **CDF and PDF Improvement**
   - Replaced `scipy.stats.norm.cdf()` with `scipy.special.ndtr()` for faster CDF calculations

## Performance Results

### Benchmark (Original) Model
- **Execution Time**: ~45 seconds

### Optimized Model
- **Average Execution Time**: ~0.0136 seconds

### Speed Improvement
- **Speedup**: Approximately **1,800x faster**
- **Time Reduction**: From 45 seconds to 0.0136 seconds

The optimized model processes the same dataset in a fraction of a second compared to the original 45-second execution time, making it suitable for real-time or high-frequency options pricing applications.

## Further Improvements

### 9. **Numba JIT Compilation**
   - Apply `@numba.jit` decorator to critical computational functions
   - Compile NumPy operations to machine code for near-C performance
   - Can provide additional 2-10x speedup for numerical computations

### 10. **Solver Optimization**
   - Further optimize the Newton-Raphson root-finding algorithm
   - Implement adaptive step sizing and convergence criteria

### 11. **Rational Approximate Initial Guess**
   - Can provide more accurate starting points, further reducing solver iterations
   - Li, Minqiang. "Approximate inversion of the Black–Scholes formula using rational functions." European Journal of Operational Research 185.2 (2008): 743-759.


