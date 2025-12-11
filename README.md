# Barrier Option Pricing Numerical Simulation Project

## Project Overview

This project provides a comprehensive Monte Carlo simulation-based library for pricing barrier options, supporting two stochastic process models: Geometric Brownian Motion (GBM) and the 3/2 stochastic volatility model. Barrier options are path-dependent exotic options whose payoff depends on whether the underlying asset price reaches a predetermined barrier level during the option's lifetime. This project is coauthored by Jinsui Liu, Youxuan Li and Zhengyuan Huang.

## Project Structure

```
/
├── GPU_version
│   ├── all_content_gpu.py         
│   └── utils_gpu.py               
├── README.md              # Project documentation
├── all_content.py         # Main implementation and analysis code
├── utils.py               # Core function library
└── user manual.md         # Detailed user manual
```

## Key Features

- **Support for 8 Barrier Option Types**: Including up/down knock-out/knock-in call/put options
- **Two Stochastic Process Models**:
  - Geometric Brownian Motion (GBM) - Constant volatility
  - 3/2 Stochastic Volatility Model - More realistic volatility dynamics
- **Monte Carlo Simulation**: Efficient vectorized implementation
- **Confidence Interval Estimation**: Provides 99% confidence intervals for pricing accuracy assessment
- **Sensitivity Analysis**: Comprehensive analysis of key parameter sensitivities

## Supported Barrier Option Types

| Option Type       | Code | Description                                                                 |
| ----------------- | ---- | --------------------------------------------------------------------------- |
| Up-and-out call   | 0    | Call option that expires worthless if underlying price hits upper barrier   |
| Down-and-out call | 1    | Call option that expires worthless if underlying price hits lower barrier   |
| Up-and-out put    | 2    | Put option that expires worthless if underlying price hits upper barrier    |
| Down-and-out put  | 3    | Put option that expires worthless if underlying price hits lower barrier    |
| Up-and-in call    | 4    | Call option that becomes active only if underlying price hits upper barrier |
| Down-and-in call  | 5    | Call option that becomes active only if underlying price hits lower barrier |
| Up-and-in put     | 6    | Put option that becomes active only if underlying price hits upper barrier  |
| Down-and-in put   | 7    | Put option that becomes active only if underlying price hits lower barrier  |

## Core Functions

### Path Simulation Functions

- [`sim_GBM(r, sigma, S0, T, N, M)`](utils.py:19): Simulates stock price paths using Geometric Brownian Motion
- [`sim_3over2(r, theta, kappa, lbd, rho, S0, V0, T, N, M)`](utils.py:65): Simulates stock price and volatility paths using the 3/2 stochastic volatility model

### Option Pricing Functions

- [`sim_option_with_CI(S_paths, K, B, r, T, option_type=0)`](utils.py:259): Calculates barrier option price with 99% confidence interval
- [`sim_option(S_paths, K, B, r, T, option_type=0)`](utils.py:205): Calculates mean barrier option price
- [`sim_options(S_paths, K, B, r, T, option_type=0)`](utils.py:137): Returns individual payoffs for each simulation path

## Quick Start

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

## Analysis Content

The project includes the following sensitivity analyses:

1. **Sensitivity to Number of Paths (M)**: Evaluates Monte Carlo simulation convergence
2. **Sensitivity to Number of Time Steps (N)**: Assesses discretization error impact
3. **Sensitivity to 3/2 Model Parameter λ**: Evaluates volatility-of-volatility parameter influence
4. **Sensitivity to Correlation Coefficient ρ**: Assesses price-volatility correlation impact

## Mathematical Background

### Geometric Brownian Motion
The stock price follows the stochastic differential equation:
```
dS_t = r * S_t * dt + σ * S_t * dW_t
```

### 3/2 Stochastic Volatility Model
The variance follows:
```
dV_t = κ * V_t * (θ - V_t) * dt + λ * V_t^(3/2) * dB_t
dS_t = r * S_t * dt + sqrt(V_t) * S_t * dW_t
```

### Barrier Option Payoff
For each path, the payoff is calculated as:
```
Payoff = max(0, S_T - K) * exp(-r*T) * I(valid_path)  # for call options
Payoff = max(0, K - S_T) * exp(-r*T) * I(valid_path)  # for put options
```
where I(valid_path) is an indicator function that equals 1 if the path satisfies the barrier condition, 0 otherwise.

## Dependencies

- NumPy
- NumPy.random
- pandas (for data analysis)
- matplotlib (for plotting)
- tqdm (for progress bars)

## Installation and Setup

1. Ensure all dependencies are installed:
```bash
pip install numpy pandas matplotlib tqdm
```

2. Run Jupyter Notebook for analysis:
```bash
jupyter notebook implemtation.ipynb
```

3. Or use Python script directly:
```bash
python utils.py
```

## Accuracy and Performance

- Monte Carlo method accuracy improves with the square root of the number of paths (M)
- Confidence intervals provide a measure of estimation uncertainty
- Time step size (N) affects discretization error
- Recommended: M ≥ 10,000 paths for reasonable accuracy

## Important Notes

- Barrier conditions are checked at discrete time points only
- For more accurate results with continuous barrier monitoring, use smaller time steps (larger N)
- The 3/2 model provides more realistic volatility dynamics compared to constant volatility models
- Confidence intervals assume normal distribution of the estimator (valid for large M)