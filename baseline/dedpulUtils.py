import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


def rolling_apply(diff, k_neighbours):
    s = pd.Series(diff)
    s = np.concatenate((
                        s.iloc[:2*k_neighbours].expanding(center=False).median()[::2].values,
                        s.rolling(k_neighbours*2+1, center=True).median().dropna().values,
                        np.flip(np.flip(s.iloc[-2*k_neighbours:], axis=0).expanding(center=False).median()[::2], axis=0).values
    ))
    return s


def compute_log_likelihood(preds, kde, kde_outer_fun=lambda kde, x: kde(x)):
    likelihood = np.apply_along_axis(lambda x: kde_outer_fun(kde, x), 0, preds)
    return np.log(likelihood).mean()


def maximize_log_likelihood(preds, kde_inner_fun, kde_outer_fun, n_folds=5, kde_type='kde', bw_low=0.01, bw_high=0.4,
                            n_gauss_low=1, n_gauss_high=50, bins_low=20, bins_high=250, n_steps=25):
    kf = KFold(n_folds, shuffle=True)
    idx_best, like_best = 0, 0
    bws = np.exp(np.linspace(np.log(bw_low), np.log(bw_high), n_steps))
    n_gauss = np.linspace(n_gauss_low, n_gauss_high, n_steps).astype(int)
    bins = np.linspace(bins_low, bins_high, n_steps).astype(int)
    for idx, (bw, n_g, bin) in enumerate(zip(bws, n_gauss, bins)):
        like = 0
        for train_idx, test_idx in kf.split(preds):
            if kde_type == 'kde':
                kde = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[train_idx]), bw)
            elif kde_type == 'GMM':
                GMM = GaussianMixture(n_g, covariance_type='spherical').fit(
                    np.apply_along_axis(kde_inner_fun, 0, preds[train_idx]).reshape(-1, 1))
                kde = lambda x: np.exp(GMM.score_samples(x.reshape(-1, 1)))
            elif kde_type == 'hist':
                bars = np.histogram(preds[train_idx], bins=bin, range=(0, 1), density=True)[0]
                kde = lambda x: bars[np.clip((x // (1 / bin)).astype(int), 0, bin - 1)]
                kde_outer_fun = lambda kde, x: kde(x)

            like += compute_log_likelihood(preds[test_idx], kde, kde_outer_fun)
        if like > like_best:
            like_best, idx_best = like, idx
    if kde_type == 'kde':
        return bws[idx_best]
    elif kde_type == 'GMM':
        return n_gauss[idx_best]
    elif kde_type == 'hist':
        return bins[idx_best]


class MonotonizingTrends:
    def __init__(self, a=None, MT_coef=1):
        self.counter = dict()
        self.array_new = []
        if a is None:
            self.array_old = []
        else:
            self.add_array(a)
        self.MT_coef = MT_coef

    def add_array(self, a):
        if isinstance(a, np.ndarray) or isinstance(a, pd.Series):
            a = a.tolist()
        self.array_old = a

    def reset(self):
        self.counter = dict()
        self.array_old = []
        self.array_new = []

    def get_highest_point(self):
        if self.counter:
            return max(self.counter)
        else:
            return np.NaN

    def add_point_to_counter(self, point):
        if point not in self.counter.keys():
            self.counter[point] = 1

    def change_counter_according_to_point(self, point):
        for key in self.counter.keys():
            if key <= point:
                self.counter[key] += 1
            else:
                self.counter[key] -= self.MT_coef

    def clear_counter(self):
        for key, value in list(self.counter.items()):
            if value <= 0:
                self.counter.pop(key)

    def update_counter_with_point(self, point):
        self.change_counter_according_to_point(point)
        self.clear_counter()
        self.add_point_to_counter(point)

    def monotonize_point(self, point=None):
        if point is None:
            point = self.array_old.pop(0)
        new_point = max(point, self.get_highest_point())
        self.array_new.append(new_point)
        self.update_counter_with_point(point)
        return new_point

    def monotonize_array(self, a=None, reset=False, decay_MT_coef=False):
        if a is not None:
            self.add_array(a)
        decay_by = 0
        if decay_MT_coef:
            decay_by = self.MT_coef / len(a)

        for _ in range(len(self.array_old)):
            self.monotonize_point()
            if decay_MT_coef:
                self.MT_coef -= decay_by

        if not reset:
            return self.array_new
        else:
            array_new = self.array_new[:]
            self.reset()
            return array_new
