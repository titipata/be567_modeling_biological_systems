import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr
from scipy import special
from collections import Counter
from scipy.stats import pearsonr
from collections import namedtuple
from sklearn.metrics import normalized_mutual_info_score


Parameters = namedtuple('Parameters',
                        ['a1', 'b1', 'am1', 'bm1',
                         'a2', 'a3', 'a4', 'a5', 'a6',
                         'b2', 'b3', 'b4', 'b5', 'b6',
                         'b7', 'b8', 'b9'])


def weighted_choice(w):
    """random choice from probability distribution in list or array

    e.g. weighted_choice([0.7, 0.3]) >> output choice 0 or 1 in this case
    """
    w = np.array(w)
    w_cum = np.cumsum(w)
    throw = np.random.rand() * w_cum[-1]
    return np.searchsorted(w_cum, throw)


def sample_interval(t, samp_interval=5, n_samples=None):
    """
    Given array of t, return index with given interval

    Example
    =======
    sample_interval([0.5, 1.1, 1.4, 1.9, 2.1], samp_interval=1, n_samples=2) >> [1, 4]
    """
    if n_samples is None:
        n_samples = int(t.max() / samp_interval)

    idx = []
    for i in range(n_samples):
        idx.append(np.searchsorted(t, (i+1) * samp_interval, side='left'))
    return np.array(idx)


def gillespie_iteration(g, params):
    """
    Perform Gillespie algorithm for one iteration
    """
    a_1 = params.a1 * (g.P1 * g.G2)
    a_2 = params.am1 * g.P1G2
    a_3 = params.a1 * (g.P2 * g.G1)
    a_4 = params.bm1 * g.P2G1
    a_5 = params.a2 * g.G1
    a_6 = params.a3 * g.P2G1
    a_7 = params.a4 * g.M1
    a_8 = params.a5 * g.M1
    a_9 = params.a6 * g.P1
    a_10 = params.b2 * g.G2
    a_11 = params.b3 * g.P1G2
    a_12 = params.b4 * g.M2
    a_13 = params.b5 * g.M2
    a_14 = params.b6 * g.P2
    a_15 = params.b8 * (g.P1 * g.M2)
    a_16 = params.b7 * g.M2P1
    a_17 = params.b9 * g.M2P1
    a = [a_1, a_2, a_3, a_4, a_5,
         a_6, a_7, a_8, a_9, a_10,
         a_11, a_12, a_13, a_14, a_15,
         a_16, a_17]
    action = weighted_choice(a)
    a0 = np.sum(a)
    tau = - (1 / a0) * np.log(np.random.rand())

    if action == 0:
        g.P1 -= 1
        g.G2 -= 1
        g.P1G2 += 1
    elif action == 1:
        g.P1 += 1
        g.G2 += 1
        g.P1G2 -= 1
    elif action == 2:
        g.P2 -= 1
        g.G1 -= 1
        g.P2G1 += 1
    elif action == 3:
        g.P2 += 1
        g.G1 += 1
        g.P2G1 -= 1
    elif action == 4:
        g.M1 += 1
    elif action == 5:
        g.M1 += 1
    elif action == 6:
        g.P1 += 1
    elif action == 7:
        g.M1 -= 1
    elif action == 8:
        g.P1 -= 1
    elif action == 9:
        g.M2 += 1
    elif action == 10:
        g.M2 += 1
    elif action == 11:
        g.P2 += 1
    elif action == 12:
        g.M2 -= 1
    elif action == 13:
        g.P2 -= 1
    elif action == 14:
        g.P1 -= 1
        g.M2 -= 1
        g.M2P1 += 1
    elif action == 15:
        g.P1 += 1
        g.M2 += 1
        g.M2P1 -= 1
    elif action == 16:
        g.M2P1 -= 1
    return g, tau

if __name__ == __main__:
    """
    Running simulation
    """
    T = 0
    n_iters = 20000
    omega = 300
    params = Parameters(0.028, 0.028, 0.01, 0.01,
                        1.25, 0.75, 10/omega, 10/omega, 1/omega,
                        1.25, 0.75, 10/omega, 10/omega, 1/omega,
                        1/omega, 57/omega, 0.5/omega)
    experiments = []
    for _ in range(n_iters):
        g, tau = gillespie_iteration_full(g, params)
        T = T + tau
        experiments.append([T] + list(g.as_matrix()))
    experiments = np.array(experiments)
    T_max = experiments[:, 0].max()
    print(T_max)

    idx = sample_interval(experiments[:, 0],
                          samp_interval=1) # sample time
    experiments_samp = experiments[idx, :]
    cols = ['T'] + list(g.keys())
    experiments_samp = pd.DataFrame(experiments_samp, columns=cols)

    plt.plot(experiments_samp['T'], experiments_samp['P1'])
    plt.plot(experiments_samp['T'], experiments_samp['P2'], c='r')
    plt.legend(['P1 (repressor)', 'P2 (targeted protein)'], bbox_to_anchor=(1.04,0.9))
    plt.show()
