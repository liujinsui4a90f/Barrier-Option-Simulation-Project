# Monte Carlo Barrier Option Pricing Library

## Overview

This library provides a comprehensive implementation for pricing barrier options using Monte Carlo simulation methods. Barrier options are path-dependent exotic options whose payoff depends on whether the underlying asset's price reaches a predetermined barrier level during the option's life.

## Features

- Support for 8 different types of barrier options
- Two stochastic processes: Geometric Brownian Motion (GBM) and 3/2 stochastic volatility model
- Confidence interval estimation for pricing accuracy
- Efficient vectorized implementation using NumPy

## Barrier Option Types

The library supports the following barrier option types:

| Type | Code | Description |
|------|------|-------------|
| Up-and-out call | 0 | Call option that expires worthless if underlying price hits upper barrier |
| Down-and-out call | 1 | Call option that expires worthless if underlying price hits lower barrier |
| Up-and-out put | 2 | Put option that expires worthless if underlying price hits upper barrier |
| Down-and-out put | 3 | Put option that expires worthless if underlying price hits lower barrier |
| Up-and-in call | 4 | Call option that becomes active only if underlying price hits upper barrier |
| Down-and-in call | 5 | Call option that becomes active only if underlying price hits lower barrier |
| Up-and-in put | 6 | Put option that becomes active only if underlying price hits upper barrier |
| Down-and-in put | 7 | Put option that becomes active only if underlying price hits lower barrier |

## Functions

### `sim_GBM(r, sigma, S0, T, N, M)`
Simulates stock price paths using Geometric Brownian Motion model.

**Parameters:**
- `r`: Risk-free interest rate
- `sigma`: Volatility parameter
- `S0`: Initial stock price
- `T`: Time to maturity
- `N`: Number of time steps
- `M`: Number of simulation paths

**Returns:**
- `t_grid`: Time grid
- `S_paths`: Simulated stock price paths

### `sim_3over2(r, theta, kappa, lbd, rho, S0, V0, T, N, M)`
Simulates stock price and variance paths using the 3/2 stochastic volatility model.

**Parameters:**
- `r`: Risk-free interest rate
- `theta`: Long-term mean of variance
- `kappa`: Mean reversion speed
- `lbd`: Volatility of volatility
- `rho`: Correlation between price and volatility
- `S0`: Initial stock price
- `V0`: Initial variance
- `T`: Time to maturity
- `N`: Number of time steps
- `M`: Number of simulation paths

**Returns:**
- `t_grid`: Time grid
- `V_paths`: Simulated variance paths
- `S_paths`: Simulated stock price paths

### `sim_option_with_CI(S_paths, K, B, r, T, option_type=0)`
Calculates barrier option price with 99% confidence interval using Monte Carlo simulation.

**Parameters:**
- `S_paths`: Simulated stock price paths
- `K`: Strike price
- `B`: Barrier level
- `r`: Risk-free interest rate
- `T`: Time to maturity
- `option_type`: Integer code for option type (0-7)

**Returns:**
- `mean_price`: Estimated option price
- `CI`: 99% confidence interval [lower, upper]

### `sim_option(S_paths, K, B, r, T, option_type=0)`
Returns the mean barrier option price across all simulation paths.

### `sim_options(S_paths, K, B, r, T, option_type=0)`
Returns individual payoffs for each simulation path.

## Usage Examples

### Basic Usage

```python
import numpy as np
from utils import sim_GBM, sim_option_with_CI, option_types

# Simulate stock price paths
t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=1024, M=10000)

# Price an Up-and-in call option
barrier_level = 120
strike_price = 100
option_type = 4  # Up-and-in call

price, confidence_interval = sim_option_with_CI(
    S_paths=S_paths,
    K=strike_price,
    B=barrier_level,
    r=0.05,
    T=1,
    option_type=option_type
)

print(f"Option Price: {price:.4f}")
print(f"99% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
```

### Using Different Option Types
```python
# Down-and-out put option
down_and_out_put = sim_option_with_CI(S_paths, 100, 80, 0.05, 1, option_types['Down-and-out put'])

# Up-and-in call option
up_and_in_call = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, option_types['Up-and-in call'])
```

### Using 3/2 Stochastic Volatility Model
```python
from utils import sim_3over2

# Simulate with stochastic volatility
t_grid, V_paths, S_paths = sim_3over2(
    r=0.05, theta=0.04, kappa=2.0, lbd=0.1, rho=-0.7,
    S0=100, V0=0.04, T=1, N=1024, M=10000
)

# Price option using these paths
price, ci = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, 4)
```

## Mathematical Background

### Geometric Brownian Motion
The stock price follows the stochastic differential equation:
$$
dS_t = r  S_t dt + \sigma S_t dW_t
$$

### 3/2 Stochastic Volatility Model
The variance follows:
$$
dV_t = \kappa  V_t (\theta - V_t) dt + \lambda V_t^{3/2} dB_t\\
dS_t = r S_t dt + \sqrt{V_t} S_t dW_t
$$

### Barrier Option Payoff
For each path, the payoff is calculated as:
```
Payoff = max(0, S_T - K) * exp(-r*T) * I(valid_path)  # for calls
Payoff = max(0, K - S_T) * exp(-r*T) * I(valid_path)  # for puts
```
where I(valid_path) is an indicator function that equals 1 if the path satisfies the barrier condition, 0 otherwise.

## Accuracy and Performance

- The accuracy of the Monte Carlo method improves with the square root of the number of paths (M)
- Confidence intervals provide a measure of estimation uncertainty
- The time step size (N) affects the discretization error
- Recommended: M ≥ 10,000 paths for reasonable accuracy

## Dependencies

- NumPy
- NumPy.random

## Notes

- The barrier condition is checked at discrete time points only
- For more accurate results with continuous barrier monitoring, use smaller time steps (larger N)
- The 3/2 model provides more realistic volatility dynamics compared to constant volatility models
- Confidence intervals assume normal distribution of the estimator (valid for large M)
```