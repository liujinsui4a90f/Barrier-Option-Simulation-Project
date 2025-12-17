# Barrier Option Pricing Numerical Simulation Project

[English](#english) | [中文](#中文)

---

<a name="english"></a>
# English Version

## Project Overview

This project provides a comprehensive Monte Carlo simulation-based library for pricing barrier options, supporting two stochastic process models: Geometric Brownian Motion (GBM) and the 3/2 stochastic volatility model. Barrier options are path-dependent exotic options whose payoff depends on whether the underlying asset price reaches a predetermined barrier level during the option's lifetime.

The library is designed for both academic and practical use, offering efficient vectorized implementations in NumPy and optional GPU acceleration via PyTorch. It includes confidence interval estimation, sensitivity analysis tools, and a complete suite of eight barrier option types.

**Authors:** Jinsui Liu, Youxuan Li, Zhengyuan Huang

## Key Features

- **Support for 8 Barrier Option Types**: Up/down knock-out/knock-in call/put options.
- **Two Stochastic Process Models**:
  - Geometric Brownian Motion (GBM) – constant volatility.
  - 3/2 Stochastic Volatility Model – more realistic volatility dynamics with mean reversion and volatility-of-volatility.
- **Monte Carlo Simulation**: Highly vectorized, efficient implementations using NumPy.
- **GPU Acceleration**: Optional PyTorch-based GPU version for massive parallel simulations (see `GPU_version/`).
- **Confidence Interval Estimation**: 99% confidence intervals for pricing accuracy assessment.
- **Sensitivity Analysis**: Built-in functions to analyze sensitivity to model parameters, number of paths, time steps, etc.
- **Comprehensive Analysis Scripts**: Ready-to-run scripts for convergence studies, parameter sweeps, and result visualization.

## Supported Barrier Option Types

Each option type is encoded with a 3‑bit integer (0–7) for easy reference.

| Option Type       | Code | Description                                                                 |
|-------------------|------|-----------------------------------------------------------------------------|
| Up‑and‑out call   | 0    | Call option that expires worthless if underlying price hits upper barrier   |
| Down‑and‑out call | 1    | Call option that expires worthless if underlying price hits lower barrier   |
| Up‑and‑out put    | 2    | Put option that expires worthless if underlying price hits upper barrier    |
| Down‑and‑out put  | 3    | Put option that expires worthless if underlying price hits lower barrier    |
| Up‑and‑in call    | 4    | Call option that becomes active only if underlying price hits upper barrier |
| Down‑and‑in call  | 5    | Call option that becomes active only if underlying price hits lower barrier |
| Up‑and‑in put     | 6    | Put option that becomes active only if underlying price hits upper barrier  |
| Down‑and‑in put   | 7    | Put option that becomes active only if underlying price hits lower barrier  |

A dictionary `option_types` mapping descriptive names to codes is provided in `utils.py`.

## Project Structure

```
/
├── GPU_version/                    # GPU‑accelerated implementation (PyTorch)
│   ├── all_content_gpu.py         # Main analysis scripts for GPU
│   └── utils_gpu.py               # Core GPU functions
├── all_content.py                 # Main analysis scripts (CPU)
├── utils.py                       # Core library functions (CPU)
├── README.md                      # This document
└── user manual.md                 # Detailed function reference
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install numpy scipy pandas matplotlib tqdm
```

For GPU support (optional):

```bash
pip install torch
```

## Quick Start

### Basic Usage (GBM Model)

```python
import numpy as np
from utils import sim_GBM, sim_option_with_CI, option_types

# Simulate stock price paths
t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=1024, M=10000)

# Price an Up‑and‑in call option
barrier_level = 120
strike_price = 100
option_type = 4  # Up‑and‑in call

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
# Down‑and‑out put option
price, ci = sim_option_with_CI(S_paths, 100, 80, 0.05, 1, option_types['Down‑and‑out put'])

# Up‑and‑in call option
price, ci = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, option_types['Up‑and‑in call'])
```

### Using the 3/2 Stochastic Volatility Model

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

### GPU Acceleration

```python
from GPU_version.utils_gpu import sim_GBM, sim_option_with_CI

# The same API, but runs on GPU if CUDA is available
t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=1024, M=100000)
price, ci = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, 4)
```

## Core Functions API

### Path Simulation

- `sim_GBM(r, sigma, S0, T, N, M)` – Simulates stock price paths using Geometric Brownian Motion.
- `sim_3over2(r, theta, kappa, lbd, rho, S0, V0, T, N, M)` – Simulates stock price and variance paths using the 3/2 stochastic volatility model.

### Option Pricing

- `sim_options(S_paths, K, B, r, T, option_type=0)` – Returns individual payoffs for each simulation path.
- `sim_option(S_paths, K, B, r, T, option_type=0)` – Returns the mean barrier option price.
- `sim_option_with_CI(S_paths, K, B, r, T, option_type=0)` – Returns the mean price and a 99% confidence interval.

### Utility

- `option_types` – Dictionary mapping descriptive names to integer codes (0–7).
- `find_h(mu, sigma, T=1.0, target=0.5)` – Solves for barrier width given a survival probability (used in theoretical calculations).

Refer to `user manual.md` for detailed parameter descriptions and examples.

## Mathematical Background

### Geometric Brownian Motion

The stock price follows the stochastic differential equation:

```
dS_t = r·S_t·dt + σ·S_t·dW_t
```

### 3/2 Stochastic Volatility Model

The variance follows:

```
dV_t = κ·V_t·(θ - V_t)·dt + λ·V_t^(3/2)·dB_t
dS_t = r·S_t·dt + √V_t·S_t·dW_t
```

where `W_t` and `B_t` are correlated Brownian motions with correlation `ρ`.

### Barrier Option Payoff

For each simulated path, the discounted payoff is:

```
Payoff = max(0, S_T - K)·exp(-r·T)·I(valid_path)   # for call options
Payoff = max(0, K - S_T)·exp(-r·T)·I(valid_path)   # for put options
```

`I(valid_path)` is an indicator that equals 1 if the path satisfies the barrier condition, 0 otherwise.

## Analysis Content

The project includes ready‑to‑run sensitivity analyses:

1. **Sensitivity to Number of Paths (M)** – Evaluates Monte Carlo convergence.
2. **Sensitivity to Number of Time Steps (N)** – Assesses discretization error impact.
3. **Sensitivity to 3/2 Model Parameter λ** – Examines volatility‑of‑volatility influence.
4. **Sensitivity to Correlation Coefficient ρ** – Studies price‑volatility correlation impact.

Run `python all_content.py` to generate all analysis figures and tables (results are saved in the `results/` folder). For GPU‑based analyses, use `python GPU_version/all_content_gpu.py`.

## Dependencies

### Required
- `numpy` (>=1.19)
- `scipy` (>=1.5)
- `pandas` (>=1.2)
- `matplotlib` (>=3.3)
- `tqdm` (>=4.50)

### Optional (GPU version)
- `torch` (>=1.9) with CUDA support

All dependencies can be installed via 
```
pip install -r requirements.txt
``` 
or manually as shown above.

---

<a name="中文"></a>
# 中文版本

## 项目概述

本项目提供了一个基于蒙特卡洛模拟的障碍期权定价库，支持两种随机过程模型：几何布朗运动（GBM）和3/2随机波动率模型。障碍期权是路径依赖的奇异期权，其收益取决于标的资产价格在期权存续期内是否达到预设的障碍水平。

该库设计用于学术和实际应用，提供高效的NumPy向量化实现，并支持通过PyTorch进行GPU加速。它包括置信区间估计、敏感性分析工具以及完整的八种障碍期权类型。

**作者：** Jinsui Liu, Youxuan Li, Zhengyuan Huang

## 主要特性

- **支持8种障碍期权类型**：向上/向下敲出/敲入看涨/看跌期权。
- **两种随机过程模型**：
  - 几何布朗运动（GBM）——常数波动率。
  - 3/2随机波动率模型——具有均值回归和波动率之波动率的更真实波动率动态。
- **蒙特卡洛模拟**：使用NumPy实现的高度向量化、高效的计算。
- **GPU加速**：可选的基于PyTorch的GPU版本，用于大规模并行模拟（参见`GPU_version/`）。
- **置信区间估计**：提供99%置信区间以评估定价精度。
- **敏感性分析**：内置函数用于分析模型参数、路径数量、时间步长等的敏感性。
- **全面的分析脚本**：开箱即用的脚本，用于收敛性研究、参数扫描和结果可视化。

## 支持的障碍期权类型

每种期权类型用一个3位整数（0–7）编码以便引用。

| 期权类型         | 代码 | 描述                                                                 |
|-------------------|------|----------------------------------------------------------------------|
| 向上敲出看涨期权 | 0    | 当标的资产价格触及上障碍时失效的看涨期权                             |
| 向下敲出看涨期权 | 1    | 当标的资产价格触及下障碍时失效的看涨期权                             |
| 向上敲出看跌期权 | 2    | 当标的资产价格触及上障碍时失效的看跌期权                             |
| 向下敲出看跌期权 | 3    | 当标的资产价格触及下障碍时失效的看跌期权                             |
| 向上敲入看涨期权 | 4    | 仅当标的资产价格触及上障碍时才生效的看涨期权                         |
| 向下敲入看涨期权 | 5    | 仅当标的资产价格触及下障碍时才生效的看涨期权                         |
| 向上敲入看跌期权 | 6    | 仅当标的资产价格触及上障碍时才生效的看跌期权                         |
| 向下敲入看跌期权 | 7    | 仅当标的资产价格触及下障碍时才生效的看跌期权                         |

`utils.py`中提供了将描述性名称映射到代码的字典`option_types`。

## 项目结构

```
/
├── GPU_version/                    # GPU加速实现（PyTorch）
│   ├── all_content_gpu.py         # GPU主分析脚本
│   └── utils_gpu.py               # GPU核心函数
├── all_content.py                 # 主分析脚本（CPU）
├── utils.py                       # 核心库函数（CPU）
├── README.md                      # 本文档
└── user manual.md                 # 详细函数参考手册
```

## 安装指南

### 前提条件

- Python 3.7 或更高版本
- pip（Python包管理器）

### 安装依赖

```bash
pip install numpy scipy pandas matplotlib tqdm
```

如需GPU支持（可选）：

```bash
pip install torch
```

### 验证安装

克隆或下载项目后，运行快速测试：

```python
python -c "import numpy, scipy, pandas, matplotlib, tqdm; print('所有CPU依赖已满足')"
```

## 快速开始

### 基本用法（GBM模型）

```python
import numpy as np
from utils import sim_GBM, sim_option_with_CI, option_types

# 模拟股票价格路径
t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=1024, M=10000)

# 定价一个向上敲入看涨期权
barrier_level = 120
strike_price = 100
option_type = 4  # 向上敲入看涨期权

price, confidence_interval = sim_option_with_CI(
    S_paths=S_paths,
    K=strike_price,
    B=barrier_level,
    r=0.05,
    T=1,
    option_type=option_type
)

print(f"期权价格: {price:.4f}")
print(f"99% 置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
```

### 使用不同的期权类型

```python
# 向下敲出看跌期权
price, ci = sim_option_with_CI(S_paths, 100, 80, 0.05, 1, option_types['向下敲出看跌期权'])

# 向上敲入看涨期权
price, ci = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, option_types['向上敲入看涨期权'])
```

### 使用3/2随机波动率模型

```python
from utils import sim_3over2

# 使用随机波动率模拟
t_grid, V_paths, S_paths = sim_3over2(
    r=0.05, theta=0.04, kappa=2.0, lbd=0.1, rho=-0.7,
    S0=100, V0=0.04, T=1, N=1024, M=10000
)

# 使用这些路径定价期权
price, ci = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, 4)
```

### GPU加速

```python
from GPU_version.utils_gpu import sim_GBM, sim_option_with_CI

# 相同的API，但如果CUDA可用则运行在GPU上
t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=1024, M=100000)
price, ci = sim_option_with_CI(S_paths, 100, 120, 0.05, 1, 4)
```

## 核心函数API

### 路径模拟

- `sim_GBM(r, sigma, S0, T, N, M)` – 使用几何布朗运动模拟股票价格路径。
- `sim_3over2(r, theta, kappa, lbd, rho, S0, V0, T, N, M)` – 使用3/2随机波动率模型模拟股票价格和方差路径。

### 期权定价

- `sim_options(S_paths, K, B, r, T, option_type=0)` – 返回每个模拟路径的个体收益。
- `sim_option(S_paths, K, B, r, T, option_type=0)` – 返回障碍期权的平均价格。
- `sim_option_with_CI(S_paths, K, B, r, T, option_type=0)` – 返回平均价格和99%置信区间。

### 工具函数

- `option_types` – 将描述性名称映射到整数代码（0–7）的字典。
- `find_h(mu, sigma, T=1.0, target=0.5)` – 给定生存概率求解障碍宽度（用于理论计算）。

详细参数说明和示例请参阅`user manual.md`。

## 数学背景

### 几何布朗运动

股票价格遵循随机微分方程：

```
dS_t = r·S_t·dt + σ·S_t·dW_t
```

### 3/2随机波动率模型

方差遵循：

```
dV_t = κ·V_t·(θ - V_t)·dt + λ·V_t^(3/2)·dB_t
dS_t = r·S_t·dt + √V_t·S_t·dW_t
```

其中`W_t`和`B_t`是具有相关系数`ρ`的相关布朗运动。

### 障碍期权收益

对于每个模拟路径，贴现收益为：

```
收益 = max(0, S_T - K)·exp(-r·T)·I(有效路径)   # 看涨期权
收益 = max(0, K - S_T)·exp(-r·T)·I(有效路径)   # 看跌期权
```

`I(有效路径)`是指示函数，如果路径满足障碍条件则为1，否则为0。

## 分析内容

本项目包含开箱即用的敏感性分析：

1. **对路径数量（M）的敏感性** – 评估蒙特卡洛收敛性。
2. **对时间步数量（N）的敏感性** – 评估离散化误差影响。
3. **对3/2模型参数λ的敏感性** – 研究波动率之波动率的影响。
4. **对相关系数ρ的敏感性** – 研究价格‑波动率相关性的影响。

运行`python all_content.py`生成所有分析图表和表格（结果保存在`results/`文件夹中）。对于基于GPU的分析，使用`python GPU_version/all_content_gpu.py`。

## 依赖项

### 必需
- `numpy` (>=1.19)
- `scipy` (>=1.5)
- `pandas` (>=1.2)
- `matplotlib` (>=3.3)
- `tqdm` (>=4.50)

### 可选（GPU版本）
- `torch` (>=1.9) 并支持CUDA

所有依赖项可以通过
```
pip install -r requirements.txt
``` 
或如上所示手动安装。