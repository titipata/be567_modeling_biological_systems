import numpy as np
from collections import Counter
import random
from scipy import special
from scipy.stats import pearsonr
from math import factorial
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

"""
Simple Gillespie simulation of gene transcription
I -- lambd --> A
A -- gamma --> I
A --   mu  --> A + M
M -- delta --> 0
M --  mu_p --> M + P
P -- delta_p --> 0
"""

def weighted_choice(w):
    """random choice from probability distribution in list or array

    e.g. weighted_choice([0.7, 0.3]) >> output choice 0 or 1 in this case
    """
    if not isinstance(w, np.ndarray):
        w = np.array(w)
    w_cum = np.cumsum(w)
    w_cum = w_cum / w_cum[-1]
    throw = np.random.rand()
    return np.searchsorted(w_cum, throw)


def sample_interval(t, interval=5, n_samples=1000):
    """
    Given array of t, return index with given interval

    Example
    =======
    sample_interval([0.5, 1.1, 1.4, 1.9, 2.1],
                    interval=1, n_samples=2) >> [1, 4]
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

    tau = - (1 / a0) * np.log(np.random.rand())
    return [I, A, M, P], tau


def run_experiment(n_experiments=100000, interval=3, n_samples=1000,
                   lambd=1, gamma=1, mu=1, delta=1, mu_p=1, delta_p=1,
                   return_stats=False):
    """
    Run Gillespie experiments and record output with time interval `interval`.
    This also returns, Fano factor of mRNA and protein and correlation between
    mRNA and protein
    """
    I, A, M, P = 0, 1, 0, 0 # initial condition
    experiments = []
    T = 0

    for n_experiment in range(n_experiments):
        [I, A, M, P], tau = gillespie_iteration(I, A, M, P,
                                                lambd=lambd, gamma=gamma,
                                                mu=mu, delta=delta,
                                                mu_p=mu_p, delta_p=delta_p)
        T = T + tau
        experiments.append((T, M, P))
    experiments = np.array(experiments)
    idx = sample_interval(experiments[:, 0], interval=interval, n_samples=n_samples)

    experiments_samp = experiments[idx, :]

    fano_mrna = np.var(experiments_samp[:, 1]) / np.mean(experiments_samp[:, 1])
    fano_protein = np.var(experiments_samp[:, 2]) / np.mean(experiments_samp[:, 2])

    r, p_value = pearsonr(experiments_samp[:, 1],
                          experiments_samp[:, 2])
    if return_stats:
        return experiments_samp, (fano_mrna, fano_protein), r
    else:
        return experiments_samp


def calculate_fm(m, lambd=1, gamma=1, mu=1, delta=1):
    """
    Bionomial probability of given mRNA number `m` in
    instantaneous burst regime
    """
    t1 = special.gamma(lambd/delta + m) / (special.gamma(lambd/delta) * factorial(m))
    t2 = ((mu/gamma) / (mu/gamma + 1)) ** m
    t3 = (1 / (mu/ gamma + 1)) ** (lambd / delta)
    return t1 * t2 * t3


def calculate_expected_mean(lambd=1, gamma=1, mu=1,
                            delta=1, mu_p=1, delta_p=1):
    """
    Expected mean of mRNA and protein
    """
    mean_mrna =  (mu/delta) * (lambd / (lambd + gamma))
    mean_protein = (mu/delta) * (lambd / (lambd + gamma)) * (mu_p / delta_p)
    return mean_mrna, mean_protein


if __name__ == '__main__':
    """
    >> python simple_gillespie.py
    """
    lambd, gamma, mu, delta, mu_p, delta_p = 0.08, 0.01, 5, 0.3, 0.1, 1
    experiments = run_experiment(n_experiments=100000, interval=1,
                                 lambd=lambd, gamma=gamma, mu=mu,
                                 delta=delta, mu_p=mu_p, delta_p=delta_p)
    plt.hist(experiments[:, 1], bins=range(0, 50))
    plt.xlabel('Number of mRNA')
    plt.ylabel('Frequency')
    plt.title('Distribution of mRNA')
    plt.show()
