import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List

from utils import sim_GBM, sim_3over2, sim_options, sim_option, sim_option_with_CI, option_types

def BS_basic():
    t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=252, M=1000000)
    B = np.array([80, 85, 90, 95, 105, 110, 115, 120])

    values: Dict[str,List] = {}
    for type_str, type_int in option_types.items():
        values[type_str] = []
        for b in B:

            if (b < 100 and type_int % 2 == 0) or (b > 100 and type_int % 2 != 0):
                value = np.nan
            else:
                value = sim_option(S_paths, 100, b, 0.05, 1, type_int)
            # value = sim_option(S_paths, 100, b, 0.05, 1, type_int)
            values[type_str].append(value)

    values: pd.DataFrame = pd.DataFrame(values)
    values.index = [f"B={b}" for b in B]
    values = values.T
    print(values.to_latex())
    with open('results/basic_BS.tex', 'w', encoding='utf-8') as f:
        f.write(values.to_latex())

def BS_analysis_M():
    data_M_GBM = {key: {'value': np.empty(0), 'upper': np.empty(0), 'lower': np.empty(0)} for key in option_types.keys()}

    for log10_m in tqdm(np.linspace(3, 6, 30)):
        M = int(10 ** log10_m)
        t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=252, M=M)
        for type_str, type_int in option_types.items():
            b = 120 if type_int % 2 == 0 else 80
            value, CI = sim_option_with_CI(S_paths, K=100, B=b, r=0.05, T=1, option_type=type_int)

            data_M_GBM[type_str]['value'] = np.append(data_M_GBM[type_str]['value'], value)
            data_M_GBM[type_str]['upper'] = np.append(data_M_GBM[type_str]['upper'], CI[0])
            data_M_GBM[type_str]['lower'] = np.append(data_M_GBM[type_str]['lower'], CI[1])
        del t_grid
        del S_paths

    fig, axs = plt.subplots(2,4, figsize=(21, 7), dpi=300)
    axs = axs.flatten()

    m = np.linspace(3, 6, 30)
    for type_str, type_int in option_types.items():
        values = np.asarray(data_M_GBM[type_str]['value'])
        upper = np.asarray(data_M_GBM[type_str]['upper'])
        lower = np.asarray(data_M_GBM[type_str]['lower'])

        axs[type_int].plot(m, values, label='Value', c='r')
        axs[type_int].fill_between(m, lower, upper, alpha=0.7, label='CI')

        axs[type_int].set_xlabel('$\\log_{10} M$')
        axs[type_int].set_ylabel('Option Value')
        axs[type_int].legend()
        axs[type_int].set_title(f'{type_str} (B={120 if type_int % 2 == 0 else 80})')

    plt.savefig('results/sensitive_M_BS.png')

def BS_analysis_N():
    log10N = np.linspace(1,3,30)
    data_N_GBM = {key: {'value': np.empty(0), 'upper': np.empty(0), 'lower': np.empty(0)} for key in option_types.keys()}


    for log10n in tqdm(log10N):
        n = int(10**log10n)
        t_grid, S_paths = sim_GBM(r=0.05, sigma=0.2, S0=100, T=1, N=n, M=100000)
        for type_str, type_int in option_types.items():
            b = 120 if type_int % 2 == 0 else 80
            value, CI = sim_option_with_CI(S_paths, K=100, B=b, r=0.05, T=1, option_type=type_int)

            data_N_GBM[type_str]['value'] = np.append(data_N_GBM[type_str]['value'], value)
            data_N_GBM[type_str]['upper'] = np.append(data_N_GBM[type_str]['upper'], CI[0])
            data_N_GBM[type_str]['lower'] = np.append(data_N_GBM[type_str]['lower'], CI[1])

        del t_grid
        del S_paths
    
    fig, axs = plt.subplots(2,4, figsize=(21, 7), dpi=300)

    axs = axs.flatten()

    for type_str, type_int in option_types.items():
        values = np.asarray(data_N_GBM[type_str]['value'])
        upper = np.asarray(data_N_GBM[type_str]['upper'])
        lower = np.asarray(data_N_GBM[type_str]['lower'])

        axs[type_int].plot(log10N, values, label='Value', c='r')
        axs[type_int].fill_between(log10N, upper, lower, alpha=0.7, label='CI')

        axs[type_int].set_xlabel('$\\log_{10} N$')
        axs[type_int].set_ylabel('Option Value')
        axs[type_int].legend()
        axs[type_int].set_title(f'{type_str} (B={120 if type_int % 2 == 0 else 80})')

    plt.savefig('results/sensitive_N_BS.png')

def sto_vol_basic():
    t_grid, _, S_paths = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=0.67, rho=-0.5, S0=100, V0=0.2, T=1, N=252, M=1000000)
    B = np.array([80, 85, 90, 95, 105, 110, 115, 120])

    values: Dict[str,List] = {}
    for type_str, type_int in option_types.items():
        values[type_str] = []
        for b in B:

            if (b < 100 and type_int % 2 == 0) or (b > 100 and type_int % 2 != 0):
                value = np.nan
            else:
                value = sim_option(S_paths, 100, b, 0.05, 1, type_int)
            # value = sim_option(S_paths, 100, b, 0.05, 1, type_int)
            values[type_str].append(value)

    values = pd.DataFrame(values)
    values.index = [f"B={b}" for b in B]
    values = values.T
    print(values.to_latex())
    with open('results/basic_3over2.tex', 'w', encoding='utf-8') as f:
        f.write(values.to_latex())

def sto_vol_analysis_M():
    data_M_GBM = {key: {'value': np.empty(0), 'upper': np.empty(0), 'lower': np.empty(0)} for key in option_types.keys()}

    for log10_m in tqdm(np.linspace(3, 6, 30)):
        M = int(10 ** log10_m)
        t_grid, _, S_paths = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=0.67, rho=-0.5, S0=100, V0=0.2, T=1, N=252, M=M)
        del _
        for type_str, type_int in option_types.items():
            b = 120 if type_int % 2 == 0 else 80
            value, CI = sim_option_with_CI(S_paths, K=100, B=b, r=0.05, T=1, option_type=type_int)

            data_M_GBM[type_str]['value'] = np.append(data_M_GBM[type_str]['value'], value)
            data_M_GBM[type_str]['upper'] = np.append(data_M_GBM[type_str]['upper'], CI[0])
            data_M_GBM[type_str]['lower'] = np.append(data_M_GBM[type_str]['lower'], CI[1])
        del t_grid
        del S_paths
        

    fig, axs = plt.subplots(2,4, figsize=(21, 7), dpi=300)
    axs = axs.flatten()

    m = np.linspace(3, 6, 30)
    for type_str, type_int in option_types.items():
        values = np.asarray(data_M_GBM[type_str]['value'])
        upper = np.asarray(data_M_GBM[type_str]['upper'])
        lower = np.asarray(data_M_GBM[type_str]['lower'])

        axs[type_int].plot(m, values, label='Value', c='r')
        axs[type_int].fill_between(m, lower, upper, alpha=0.7, label='CI')

        axs[type_int].set_xlabel('$\\log_{10} M$')
        axs[type_int].set_ylabel('Option Value')
        axs[type_int].legend()
        axs[type_int].set_title(f'{type_str} (B={120 if type_int % 2 == 0 else 80})')

    plt.savefig('results/sensitive_M_3over2.png')

def sto_vol_analysis_N():
    log10N = np.linspace(1,3,30)
    data_N_GBM = {key: {'value': np.empty(0), 'upper': np.empty(0), 'lower': np.empty(0)} for key in option_types.keys()}


    for log10n in tqdm(log10N):
        n = int(10**log10n)
        t_grid, _, S_paths = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=0.67, rho=-0.5, S0=100, V0=0.2, T=1, N=n, M=100000)
        del t_grid
        del _
        for type_str, type_int in option_types.items():
            b = 120 if type_int % 2 == 0 else 80
            value, CI = sim_option_with_CI(S_paths, K=100, B=b, r=0.05, T=1, option_type=type_int)

            data_N_GBM[type_str]['value'] = np.append(data_N_GBM[type_str]['value'], value)
            data_N_GBM[type_str]['upper'] = np.append(data_N_GBM[type_str]['upper'], CI[0])
            data_N_GBM[type_str]['lower'] = np.append(data_N_GBM[type_str]['lower'], CI[1])

        
        del S_paths
    
    fig, axs = plt.subplots(2,4, figsize=(21, 7), dpi=300)

    axs = axs.flatten()

    for type_str, type_int in option_types.items():
        values = np.asarray(data_N_GBM[type_str]['value'])
        upper = np.asarray(data_N_GBM[type_str]['upper'])
        lower = np.asarray(data_N_GBM[type_str]['lower'])

        axs[type_int].plot(log10N, values, label='Value', c='r')
        axs[type_int].fill_between(log10N, upper, lower, alpha=0.7, label='CI')

        axs[type_int].set_xlabel('$\\log_{10} N$')
        axs[type_int].set_ylabel('Option Value')
        axs[type_int].legend()
        axs[type_int].set_title(f'{type_str} (B={120 if type_int % 2 == 0 else 80})')

    plt.savefig('results/sensitive_N_3over2.png')

def sto_vol_analysis_lambda():
    data_lbd_3over2 = {key: {'value': np.empty(0), 'upper': np.empty(0), 'lower': np.empty(0)} for key in option_types.keys()}

    for lbd in tqdm(np.linspace(0.01, 1, 100)):
        t_grid, _, S_paths = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=lbd, rho=-0.5, S0=100, V0=0.2, T=1, N=252, M=100000)
        del t_grid
        del _
        for type_str, type_int in option_types.items():
            b = 120 if type_int % 2 == 0 else 80
            value, CI = sim_option_with_CI(S_paths, K=100, B=b, r=0.05, T=1, option_type=type_int)

            data_lbd_3over2[type_str]['value'] = np.append(data_lbd_3over2[type_str]['value'], value)
            data_lbd_3over2[type_str]['upper'] = np.append(data_lbd_3over2[type_str]['upper'], CI[0])
            data_lbd_3over2[type_str]['lower'] = np.append(data_lbd_3over2[type_str]['lower'], CI[1])

    fig, axs = plt.subplots(2,4, figsize=(21, 7), dpi=300)

    axs = axs.flatten()

    lbd = np.linspace(0.01, 1, 100)
    for type_str, type_int in option_types.items():
        values = np.asarray(data_lbd_3over2[type_str]['value'])
        upper = np.asarray(data_lbd_3over2[type_str]['upper'])
        lower = np.asarray(data_lbd_3over2[type_str]['lower'])

        b1, b0 = np.polyfit(lbd, values, deg=1)
        axs[type_int].plot(lbd, b0 + b1 * lbd, c="#4bd34b", label=f'linear regression\n$y$={b1:.02f}$\\lambda$+{b0:.02f}')

        axs[type_int].plot(lbd, values, label='Value', c='r')
        axs[type_int].fill_between(lbd, lower, upper, alpha=0.7, label='CI')
        

        axs[type_int].set_xlabel('$\\lambda$')
        axs[type_int].set_ylabel('Option Value')
        axs[type_int].legend()
        axs[type_int].set_title(f'{type_str} (B={120 if type_int % 2 == 0 else 80})')

    plt.savefig('results/sensitive_lambda_3over2.png')

def sto_vol_analysis_rho():
    data_lbd_3over2 = {key: {'value': np.empty(0), 'upper': np.empty(0), 'lower': np.empty(0)} for key in option_types.keys()}

    for rho in tqdm(np.linspace(-0.99, 0.99, 100)):
        t_grid, _, S_paths = sim_3over2(r=0.05, theta=0.2, kappa=0.2, lbd=0.67, rho=rho, S0=100, V0=0.2, T=1, N=252, M=100000)
        for type_str, type_int in option_types.items():
            b = 120 if type_int % 2 == 0 else 80
            value, CI = sim_option_with_CI(S_paths, K=100, B=b, r=0.05, T=1, option_type=type_int)

            data_lbd_3over2[type_str]['value'] = np.append(data_lbd_3over2[type_str]['value'], value)
            data_lbd_3over2[type_str]['upper'] = np.append(data_lbd_3over2[type_str]['upper'], CI[0])
            data_lbd_3over2[type_str]['lower'] = np.append(data_lbd_3over2[type_str]['lower'], CI[1])

    fig, axs = plt.subplots(2,4, figsize=(21, 7), dpi=300)

    axs = axs.flatten()

    rho = np.linspace(-0.99, 0.99, 100)
    for type_str, type_int in option_types.items():
        values = np.asarray(data_lbd_3over2[type_str]['value'])
        upper = np.asarray(data_lbd_3over2[type_str]['upper'])
        lower = np.asarray(data_lbd_3over2[type_str]['lower'])

        b1, b0 = np.polyfit(rho, values, deg=1)
        axs[type_int].plot(rho, b0 + b1 * rho, c="#4bd34b", label=f'linear regression\n$y$={b1:.02f}$\\rho$+{b0:.02f}')

        axs[type_int].plot(rho, values, label='Value', c='r')
        axs[type_int].fill_between(rho, lower, upper, alpha=0.7, label='CI')

        axs[type_int].set_xlabel('$\\rho$')
        axs[type_int].set_ylabel('Option Value')
        axs[type_int].legend()
        axs[type_int].set_title(f'{type_str} (B={120 if type_int % 2 == 0 else 80})')

    plt.savefig('results/sensitive_rho_3over2.png')
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8')

    import os
    from pathlib import Path
    folder_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'results'
    folder_path.mkdir(parents=True, exist_ok=True)

    BS_basic()
    BS_analysis_M()
    BS_analysis_N()
    sto_vol_basic()
    sto_vol_analysis_M()
    sto_vol_analysis_N()
    sto_vol_analysis_lambda()
    sto_vol_analysis_rho()