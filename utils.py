#######################################
# Author: Jinsui Liu
# Date: 2025-11-24
#######################################

import numpy as np
import numpy.random as rnd
from scipy.optimize import brentq

# Dictionary mapping barrier option types to integer codes for easier reference
# Each option type is encoded using 3 bits:
# Bit 0: 0=Up barrier, 1=Down barrier
# Bit 1: 0=Call option, 1=Put option  
# Bit 2: 0=Out barrier, 1=In barrier
option_types = {'Up-and-out call': 0,     # Binary: 000
                'Down-and-out call': 1,   # Binary: 001
                'Up-and-out put': 2,      # Binary: 010
                'Down-and-out put': 3,    # Binary: 011
                'Up-and-in call': 4,      # Binary: 100
                'Down-and-in call': 5,    # Binary: 101
                'Up-and-in put': 6,       # Binary: 110
                'Down-and-in put': 7}     # Binary: 111


def sim_GBM(r, sigma, S0, T, N, M):
    """
    Simulate Geometric Brownian Motion (GBM) using Euler scheme
    
    Parameters:
    -----------
    r : float
        Drift rate (risk-free rate)
    sigma : float
        Volatility parameter
    S0 : float
        Initial stock price
    T : float
        Time to maturity
    N : int
        Number of time steps (grid intervals)
    M : int
        Number of simulation paths
    
    Returns:
    --------
    t_grid : numpy.ndarray
        Time grid from 0 to T with N+1 points
    S_paths : numpy.ndarray
        Simulated stock price paths of shape (M, N+1)
    """
    # Create uniform time grid from 0 to T with N+1 points
    t_grid = np.linspace(0, T, N+1)
    # Calculate time step size
    dt = t_grid[1] - t_grid[0]
    # Generate all random shocks: shape (M, N)
    Z = np.random.randn(M, N)
    
    # Compute multiplicative increments: 1 + r*dt + sigma*sqrt(dt)*Z
    multipliers = np.ones((M, N+1))
    multipliers[:, 1:] = 1 + r * dt + sigma * np.sqrt(dt) * Z  # shape (M, N)
    
    # Cumulative product along time axis
    S_paths = S0 * np.cumprod(multipliers, axis=1)
    
    return t_grid, S_paths


def sim_3over2(r, theta, kappa, lbd, rho, S0, V0, T, N, M):
    """
    Simulate 3/2 stochastic volatility process using Euler scheme
    
    The 3/2 model is a stochastic volatility model where variance follows:
    dV_t = kappa * V_t * (theta - V_t) * dt + lambda * V_t^(3/2) * dB_t
    
    Parameters:
    -----------
    r : float
        Drift rate (risk-free rate)
    theta : float
        Long-term mean of variance
    kappa : float
        Mean reversion speed parameter
    lbd : float
        Volatility of volatility parameter
    rho : float
        Correlation between stock price and volatility Brownian motions
    S0 : float
        Initial stock price
    V0 : float
        Initial variance
    T : float
        Time to maturity
    N : int
        Number of time steps
    M : int
        Number of simulation paths
    
    Returns:
    --------
    t_grid : numpy.ndarray
        Time grid
    V_paths : numpy.ndarray
        Simulated variance paths
    S_paths : numpy.ndarray
        Simulated stock price paths
    """
    # Create time grid
    t_grid = np.linspace(0, T, N+1)
    dt = t_grid[1] - t_grid[0]
    sqrt_dt = np.sqrt(dt)
    
    # Initialize stock price and variance paths
    S_paths = np.full((M, N+1), S0, dtype=np.float64)
    V_paths = np.full((M, N+1), V0, dtype=np.float64)

    # Calculate correlation factor for generating correlated Brownian motions
    correlation_factor = np.sqrt(1 - rho**2)
    
    # Simulate correlated paths
    for i in range(N):
        # Generate independent Brownian increments
        dW = rnd.randn(M)
        # Generate correlated Brownian increment using Cholesky decomposition
        dB = rho * dW + correlation_factor * rnd.randn(M)
        
        # Calculate square root of current variance (used in both SDEs)
        sqrt_V = np.sqrt(V_paths[:, i])
        
        # Update variance using 3/2 model formula
        # dV_t = kappa * V_t * (theta - V_t) * dt + lambda * V_t^(3/2) * dB_t
        V_paths[:, i+1] = V_paths[:, i] * (1 + kappa * (theta - V_paths[:, i]) * dt + lbd * sqrt_V * dB * sqrt_dt)
        
        # Update stock price using current variance
        # dS_t = S_t * (r * dt + sqrt(V_t) * dW_t)
        S_paths[:, i+1] = S_paths[:, i] * (1 + r * dt + sqrt_V * dW * sqrt_dt)
    
    return t_grid, V_paths, S_paths


def sim_options(S_paths, K, B, r, T, option_type: int = 0):
    """
    Simulate barrier option payoffs by Monte Carlo method (returns all path payoffs)
    
    Parameters:
    -----------
    S_paths : numpy.ndarray
        Simulated stock price paths of shape (M, N+1)
    K : float
        Strike price
    B : float
        Barrier level
    r : float
        Risk-free interest rate
    T : float
        Time to maturity
    option_type : int
        Integer code for option type (0-7)
    
    Returns:
    --------
    price : numpy.ndarray
        Payoffs for each path (array of length M)
    """
    # Decode option type using bitwise operations
    # Extract barrier direction: 0=Up, 1=Down
    is_up = (option_type & 0x01) == 0
    # Extract option type: 0=Call, 1=Put
    is_call = (option_type & 0x02) == 0
    # Extract barrier type: 0=Out, 1=In
    is_out = (option_type & 0x04) == 0
    
    # Calculate European option payoff at maturity
    if is_call:
        # Call option: max(S_T - K, 0)
        euro_option = np.maximum(0, S_paths[:, -1] - K) * np.exp(-r * T)
    else:
        # Put option: max(K - S_T, 0)
        euro_option = np.maximum(0, K - S_paths[:, -1]) * np.exp(-r * T)
        
    # Determine extreme price reached along each path
    if is_up:
        # For up barriers, find maximum price in path
        extreme_price = np.max(S_paths, axis=1)
    else:
        # For down barriers, find minimum price in path
        extreme_price = np.min(S_paths, axis=1)
        
    # Determine if each path is valid based on barrier condition
    # Logic: XOR operation determines the comparison direction
    if not (is_out ^ is_up):
        # Cases: Up-and-Out (000) or Down-and-In (011)
        # For Out options: path is valid if barrier not crossed (extreme < B)
        # For In options: path is valid if barrier crossed (extreme > B)
        # When is_out=0 and is_up=0: Up-and-Out -> extreme < B (valid)
        # When is_out=1 and is_up=1: Down-and-In -> extreme < B (valid if extreme < B is false, i.e. extreme > B)
        is_valid = extreme_price < B
    else:
        # Cases: Down-and-Out (001), Up-and-Out (010), Up-and-In (100), Down-and-In (111)
        # For other combinations
        is_valid = extreme_price > B
    
    # Apply barrier condition: payoff is 0 for invalid paths
    price = euro_option * is_valid
    
    return price


def sim_option(S_paths, K, B, r, T, option_type: int = 0):
    """
    Simulate barrier option price by Monte Carlo method (returns mean price)
    
    Parameters:
    -----------
    S_paths : numpy.ndarray
        Simulated stock price paths of shape (M, N+1)
    K : float
        Strike price
    B : float
        Barrier level
    r : float
        Risk-free interest rate
    T : float
        Time to maturity
    option_type : int
        Integer code for option type (0-7)
    
    Returns:
    --------
    mean_price : float
        Average option price across all paths
    """    
    price = sim_options(S_paths, K, B, r, T, option_type)
    
    # Return mean price across all paths
    return price.mean()


def sim_option_with_CI(S_paths, K, B, r, T, option_type: int = 0):
    """
    Simulate barrier option price with confidence interval using Monte Carlo method.
    
    This function calculates the expected discounted payoff (option price)
    and provides a 99% confidence interval for the estimate.
    
    Parameters:
    -----------
    S_paths : numpy.ndarray
        Simulated stock price paths of shape (M, N+1)
    K : float
        Strike price of the option
    B : float
        Barrier level
    r : float
        Risk-free interest rate for discounting
    T : float
        Time to maturity (in years)
    option_type : int, optional
        Integer code representing the barrier option type (default=0)
        Uses bitwise encoding as defined in option_types dictionary
    
    Returns:
    --------
    mean_price : float
        Monte Carlo estimate of the barrier option price
    CI : numpy.ndarray
        99% confidence interval [lower_bound, upper_bound] for the price estimate
    """
    price = sim_options(S_paths, K, B, r, T, option_type)
    
    # Calculate Monte Carlo estimate and 99% confidence interval
    mean_price = np.mean(price)
    # 2.575 corresponds to 99% confidence level for normal distribution
    CI = mean_price + np.array([-2.575, 2.575]) * price.std() / np.sqrt(len(price))
    
    return mean_price, CI

def find_h(mu, sigma, T=1.0, target=0.5):
    """
    Solve for h such that survival_prob(h) = target
    """
    def survival_prob(h, mu, sigma, T=1.0, n_terms=500):
        """
        Survival probability:
        P( |mu t + sigma W_t| < h for all t in [0, T] )

        Using symmetric double-barrier series expansion.
        """
        prob = 0.0
        for k in range(n_terms):
            m = 2 * k + 1
            exponent = (
                - (m ** 2) * (np.pi ** 2) * (sigma ** 2) * T / (8 * h ** 2)
            )
            drift_term = np.cosh(m * np.pi * mu * T / (2 * h))
            prob += (4 / (m * np.pi)) * np.exp(exponent) * drift_term
        return prob
    
    def objective(h):
        return survival_prob(h, mu, sigma, T) - target

    # h must be positive; choose a safe bracket
    h_min = 0.1
    h_max = 10.0

    return brentq(objective, h_min, h_max)


if __name__ == "__main__":
    # test sim_3over2
    t_grid, V_paths, S_paths = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=0.5, rho=-0.5, S0=100, V0=0.2, T=1, N=252, M=10)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(10):
        ax[0].plot(t_grid, V_paths[i], label=f"Path {i+1}")
        ax[1].plot(t_grid, S_paths[i], label=f"Path {i+1}")
        ax[0].set_title("Variance Paths")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Variance")
        ax[1].set_title("Stock Price Paths")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Stock Price")
    plt.tight_layout()
    plt.show()