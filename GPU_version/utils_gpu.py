import torch
from torch.distributions.normal import Normal
import numpy as np


option_types = {'Up-and-out call': 0,     # Binary: 000
                'Down-and-out call': 1,   # Binary: 001
                'Up-and-out put': 2,      # Binary: 010
                'Down-and-out put': 3,    # Binary: 011
                'Up-and-in call': 4,      # Binary: 100
                'Down-and-in call': 5,    # Binary: 101
                'Up-and-in put': 6,       # Binary: 110
                'Down-and-in put': 7}     # Binary: 111

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    t_grid = torch.linspace(0, T, N+1, device=device, dtype=torch.float32)
    # Calculate time step size
    dt = t_grid[1] - t_grid[0]
    # Square root of time step (used in Brownian motion increment)
    sqrt_dt = torch.sqrt(dt)
    
    # Initialize stock price paths matrix with initial value S0
    
    multiplier = torch.ones((M, N+1), device=device, dtype=torch.float32)
    multiplier[:,1:] = 1 + r * dt + sigma * sqrt_dt * torch.randn(M, N, device=device, dtype=torch.float32)
    
    S_paths = S0 * torch.cumprod(multiplier, dim=1)
    
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
    t_grid = torch.linspace(0, T, N+1, device=device, dtype=torch.float32)
    dt = t_grid[1] - t_grid[0]
    sqrt_dt = torch.sqrt(dt)
    
    # Initialize stock price and variance paths
    S_paths = torch.full((M, N+1), S0, dtype=torch.float32, device=device)
    V_paths = torch.full((M, N+1), V0, dtype=torch.float32, device=device)

    # Calculate correlation factor for generating correlated Brownian motions
    rho_tensor = torch.tensor(rho, device=device, dtype=torch.float32)
    correlation_factor = torch.sqrt(1 - rho_tensor**2)
    
    # Simulate correlated paths
    for i in range(N):
        # Generate independent Brownian increments
        dW = torch.randn(M, device=device, dtype=torch.float32)
        # Generate correlated Brownian increment using Cholesky decomposition
        dB = rho_tensor * dW + correlation_factor * torch.randn(M, device=device, dtype=torch.float32)
        
        # Calculate square root of current variance (used in both SDEs)
        sqrt_V = torch.sqrt(V_paths[:, i])
        
        # Update variance using 3/2 model formula
        # dV_t = kappa * V_t * (theta - V_t) * dt + lambda * V_t^(3/2) * dB_t
        V_paths[:, i+1] = V_paths[:, i] * (1 + kappa * (theta - V_paths[:, i]) + lbd * sqrt_V * dB * sqrt_dt)
        
        # Update stock price using current variance
        # dS_t = S_t * (r * dt + sqrt(V_t) * dW_t)
        S_paths[:, i+1] = S_paths[:, i] * (1 + r * dt + sqrt_V * dW * sqrt_dt)
    
    return t_grid, V_paths, S_paths

def sim_options(S_paths, K, B, r, T, option_type: int = 0) -> torch.Tensor:
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

    r = torch.tensor(r, dtype=torch.float32, device=device)
    T = torch.tensor(T, dtype=torch.float32, device=device)
    K = torch.tensor(K, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)
    
    # Calculate European option payoff at maturity
    if is_call:
        # Call option: max(S_T - K, 0)
        euro_option = torch.maximum(torch.zeros((), dtype=torch.float32, device=device), S_paths[:, -1] - K) * torch.exp(-r * T)
    else:
        # Put option: max(K - S_T, 0)
        euro_option = torch.maximum(torch.zeros((), dtype=torch.float32, device=device), K - S_paths[:, -1]) * torch.exp(-r * T)
        
    # Determine extreme price reached along each path
    if is_up:
        # For up barriers, find maximum price in path
        extreme_price = torch.max(S_paths, dim=1).values
    else:
        # For down barriers, find minimum price in path
        extreme_price = torch.min(S_paths, dim=1).values
        
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
    return price.mean().item()

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
    price = sim_options(S_paths, K, B, r, T, option_type).cpu().numpy()
    
    # Calculate Monte Carlo estimate and 99% confidence interval
    mean_price = price.mean()
    # 2.575 corresponds to 99% confidence level for normal distribution
    CI = mean_price + np.array([-2.575, 2.575]) * price.std() / np.sqrt(len(price))
    
    return mean_price, CI

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t, V, S = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=0.67, rho=-0.5, S0=100, V0=0.2, T=1, N=252, M=10)

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    for i in range(10):
        ax[0].plot(t.cpu().numpy(), S[i, :].cpu().numpy())
        ax[0].set_title('3/2 Model Sample Paths')
        ax[0].set_xlabel('Time')  
        ax[0].set_ylabel('Stock Price')

        ax[1].plot(t.cpu().numpy(), V[i, :].cpu().numpy())
        ax[1].set_title('3/2 Model Variance Paths')
        ax[1].set_xlabel('Time')  
        ax[1].set_ylabel('Variance')

    plt.show()