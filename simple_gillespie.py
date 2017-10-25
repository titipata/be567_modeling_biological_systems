import numpy as np
import pandas as pd
import random
from scipy.stats import linregress, pearsonr


def weighted_choice(w):
    """random choice from probability distribution in list or array

    e.g. weighted_choice([0.7, 0.3]) >> output choice 0 or 1 in this case
    """
    if not isinstance(w, np.ndarray):
        w = np.array(w)
    w_cum = np.cumsum(w)
    throw = np.random.rand() * w_cum[-1]
    return np.searchsorted(w_cum, throw)


def generate_waiting_time(a0):
    """
    Generating waiting time given the rate a0
    """
    tau = - (1 / a0) * np.log(np.random.rand())
    return tau


def find_nearest_index(t, value):
    """Find nearest index from the given value"""
    idx = (np.abs(t - value)).argmin()
    return idx


def sample_interval(t, interval=5, n_samples=1000):
    """
    Given array of t, return index with given interval

    Example
    =======
    sample_interval([0.5, 1.1, 1.4, 1.9, 2.1], interval=1, n_samples=2) >> [1, 4]
    """
    idx = []
    for i in range(n_samples):
        idx.append(np.searchsorted(t, (i+1) * interval, side='left'))
    return np.array(idx)


def gillespie_iteration(I=0, A=1, M=0, P=0,
                        lambd=1, gamma=0, mu=1, delta=1, mu_p=1, delta_p=1):
    """
    Perform Gillespie algorithm for one iteration,
    Update value of the given I, A, M, P by action
    randomly selected from throwing dart experiment
    """
    a1 = lambd * I
    a2 = gamma * A
    a3 = mu * A
    a4 = delta * M
    a5 = mu_p * M
    a6 = delta_p * P
    a = np.array([a1, a2, a3, a4, a5, a6])
    a0 = np.sum(a)
    a = a/a0
    action = weighted_choice(a)

    if action == 0:
        I = I - 1
        A = A + 1
    elif action == 1:
        I = I + 1
        A = A - 1
    elif action == 2:
        M = M + 1
    elif action == 3:
        M = M - 1
    elif action == 4:
        P = P + 1
    else:
        P = P - 1

    tau = generate_waiting_time(a0)
    return [I, A, M, P], tau

if __name__ == '__main__':

    # running multiple experiment and recording M at time T_max
    I, A, M, P = 0, 1, 0, 0
    M_list = [] # number of mRNA at the end of sim
    T_max = 10
    n_simulations = 1000
    for n_simulation in range(n_simulations):
        T = 0
        while T <= T_max:
            [I, A, M, P], tau = gillespie_iteration(I, A, M, P)
            T = T + tau
        M_list.append(M) # output

    # running for long time and recording every given interval
    experiments = []
    T = 0
    n_experiments = 100000
    for n_experiment in range(n_experiments):
        [I, A, M, P], tau = gillespie_iteration(I, A, M, P)
        T = T + tau
        experiments.append((T, M))
    experiment = np.array(experiments)
    idx = sample_interval(experiment[:, 0], interval=5, n_samples=1000)
    M_list = list(experiment[idx, 1]) # output
