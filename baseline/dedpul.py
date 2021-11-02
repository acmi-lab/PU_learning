import numpy as np
from dedpulUtils import maximize_log_likelihood, rolling_apply, MonotonizingTrends
from scipy.stats import gaussian_kde

def estimate_diff(preds, target, bw_mix=0.05, bw_pos=0.1, kde_mode='logit', threshold='mid', k_neighbours=None,
                  tune=False, MT=False, MT_coef=0.25, decay_MT_coef=False, kde_type='kde', bins_mix=20, bins_pos=20):
    """
    Estimates densities of predictions y(x) for P and U and ratio between them f_p / f_u for U sample;
        uses kernel density estimation (kde);
        post-processes difference of estimated densities - imposes monotonicity on lower preds
        (so that diff is partly non-decreasing) and applies rolling median to further reduce variance
    :param preds: predictions of NTC y(x), probability of belonging to U rather than P, np.array with shape (n,)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param bw_mix: bandwidth for kde of U
    :param bw_pos: bandwidth for kde of P
    :param kde_mode: 'prob', 'log_prob' or 'logit'; default is 'logit'
    :param monotonicity: monotonicity is imposed on density difference for predictions below this number, float in [0, 1]
    :param k_neighbours: difference is relaxed with median rolling window with size k_neighbours * 2 + 1,
        default = int(preds[target == 1].shape[0] // 10)
    :return: difference of densities f_p / f_u for U sample
    """

    if kde_mode is None:
        kde_mode = 'logit'

    if (threshold is None) or (threshold == 'mid'):
        threshold = preds[target == 1].mean() / 2 + preds[target == 0].mean() / 2
    elif threshold == 'low':
        threshold = preds[target == 0].mean()
    elif threshold == 'high':
        threshold = preds[target == 1].mean()

    if k_neighbours is None:
        k_neighbours = int(preds[target == 1].shape[0] // 20)

    if kde_mode == 'prob':
        kde_inner_fun = lambda x: x
        kde_outer_fun = lambda dens, x: dens(x)
    elif kde_mode == 'log_prob':
        kde_inner_fun = lambda x: np.log(x)
        kde_outer_fun = lambda dens, x: dens(np.log(x)) / (x + 10 ** -5)
    elif kde_mode == 'logit':
        kde_inner_fun = lambda x: np.log((x + 1e-7) / (1 - x + 10 ** -5))
        kde_outer_fun = lambda dens, x: dens(np.log((x + 1e-7) / (1 - x + 10 ** -5))) / (x * (1 - x) + 10 ** -5)

    if kde_type == 'kde':
        if tune:
            bw_mix = maximize_log_likelihood(preds[target == 1], kde_inner_fun, kde_outer_fun, kde_type=kde_type)
            bw_pos = maximize_log_likelihood(preds[target == 0], kde_inner_fun, kde_outer_fun, kde_type=kde_type)

        kde_mix = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[target == 1]), bw_mix)
        kde_pos = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[target == 0]), bw_pos)

    elif kde_type == 'hist':
        if tune:
            bins_mix = maximize_log_likelihood(preds[target == 1], kde_inner_fun, lambda kde, x: kde(x),
                                               kde_type=kde_type)
            bins_pos = maximize_log_likelihood(preds[target == 0], kde_inner_fun, lambda kde, x: kde(x),
                                               kde_type=kde_type)
        bars_mix = np.histogram(preds[target == 1], bins=bins_mix, range=(0, 1), density=True)[0]
        bars_pos = np.histogram(preds[target == 0], bins=bins_pos, range=(0, 1), density=True)[0]

        kde_mix = lambda x: bars_mix[np.clip((x // (1 / bins_mix)).astype(int), 0, bins_mix-1)]
        kde_pos = lambda x: bars_pos[np.clip((x // (1 / bins_pos)).astype(int), 0, bins_pos-1)]
        kde_outer_fun = lambda kde, x: kde(x)

    # sorting to relax and impose monotonicity
    sorted_mixed = np.sort(preds[target == 1])

    diff = np.apply_along_axis(lambda x: kde_outer_fun(kde_pos, x) / (kde_outer_fun(kde_mix, x) + 10 ** -5), axis=0,
                               arr=sorted_mixed)
    diff[diff > 50] = 50
    diff = rolling_apply(diff, 5)
    diff = np.append(
        np.flip(np.maximum.accumulate(np.flip(diff[sorted_mixed <= threshold], axis=0)), axis=0),
        diff[sorted_mixed > threshold])
    diff = rolling_apply(diff, k_neighbours)

    if MT:
        MTrends = MonotonizingTrends(MT_coef=MT_coef)
        diff = np.flip(np.array(MTrends.monotonize_array(np.flip(diff, axis=0), reset=True, decay_MT_coef=decay_MT_coef)), axis=0)

    diff.sort()
    diff = np.flip(diff, axis=0)

    # desorting
    diff = diff[np.argsort(np.argsort(preds[target == 1]))]

    return diff

def estimate_poster_em(diff=None, mode='dedpul', converge=True, tol=10**-5, max_iterations=1000, \
                       nonconverge=True, step=0.001, max_diff=0.05, disp=False,
                       alpha=None, alpha_as_mean_poster=True, **kwargs):
    """
    Performs Expectation-Maximization to estimate posteriors and priors alpha (if not provided) of N in U
        with either of 'en' or 'dedpul' methods; both 'converge' and 'nonconverge' are recommended to be set True for
        better estimate
    :param diff: difference of densities f_p/f_u for the sample U, np.array (n,), output of estimate_diff()
    :param preds: predictions of classifier, np.array with shape (n,)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param mode: 'dedpul' or 'en'; if 'dedpul', diff needs to be provided; if 'en', preds and target need to be provided
    :param converge: True or False; True if convergence estimate should be computed
    :param tol: tolerance of error between priors and mean posteriors, indicator of convergence
    :param max_iterations: if exceeded, search of converged alpha stops even if tol is not reached
    :param nonconverge: True or False; True if non-convergence estimate should be computed
    :param step: gap between points of the [0, 1, step] gird to choose best alpha from
    :param max_diff: alpha with difference of mean posteriors and priors bigger than max_diff cannot be chosen;
        an heuristic to choose bigger alpha
    :param plot: True or False, if True - plots ([0, 1, grid], mean posteriors - alpha) and
        ([0, 1, grid], second lag of (mean posteriors - alpha))
    :param disp: True or False, if True - displays if the algorithm didn't converge
    :param alpha: proportions of N in U; is estimated if None
    :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample
    """
    assert converge + nonconverge, "At least one of 'converge' and 'nonconverge' has to be set to 'True'"

    if alpha is not None:
        if mode == 'dedpul':
            alpha, poster = estimate_poster_dedpul(diff, alpha=alpha, alpha_as_mean_poster=alpha_as_mean_poster, tol=tol, **kwargs)
        return alpha, poster

    # if converge:
    alpha_converge = 0
    for i in range(max_iterations):

        if mode.endswith('dedpul'):
            _, poster_converge = estimate_poster_dedpul(diff, alpha=alpha_converge, **kwargs)

        mean_poster = np.mean(poster_converge)
        error = mean_poster - alpha_converge

        if np.abs(error) < tol:
            break
        if np.min(poster_converge) > 0:
            break
        alpha_converge = mean_poster

    if disp:
        if i >= max_iterations - 1:
            print('max iterations exceeded')

    # if nonconverge:

    errors = np.array([])
    for alpha_nonconverge in np.arange(0, 1, step):

        if mode.endswith('dedpul'):
            _, poster_nonconverge = estimate_poster_dedpul(diff, alpha=alpha_nonconverge, **kwargs)
        errors = np.append(errors, np.mean(poster_nonconverge) - alpha_nonconverge)

    idx = np.argmax(np.diff(np.diff(errors))[errors[1: -1] < max_diff])
    alpha_nonconverge = np.arange(0, 1, step)[1: -1][errors[1: -1] < max_diff][idx]


    # if converge and not nonconverge:
    #     return alpha_converge, poster_converge

    if ((alpha_nonconverge >= alpha_converge) or#converge and nonconverge and
        (((errors < 0).sum() > 1) and (alpha_converge < 1 - step))):
        return alpha_converge, poster_converge

    elif nonconverge:
        if mode == 'dedpul':
            _, poster_nonconverge = estimate_poster_dedpul(diff, alpha=alpha_nonconverge, **kwargs)

        if disp:
            print('didn\'t converge')
        return alpha_nonconverge, poster_nonconverge
        # return np.mean(poster_nonconverge), poster_nonconverge

    else:
        if disp:
            print('didn\'t converge')
        return None, None

def estimate_poster_dedpul(diff, alpha=None, quantile=0.05, alpha_as_mean_poster=False, max_it=100, **kwargs):
    """
    Estimates posteriors and priors alpha (if not provided) of N in U with dedpul method
    :param diff: difference of densities f_p / f_u for the sample U, np.array (n,), output of estimate_diff()
    :param alpha: priors, share of N in U (estimated if None)
    :param quantile: if alpha is None, relaxation of the estimate of alpha;
        here alpha is estimaeted as infinum, and low quantile is its relaxed version;
        share of posteriors probabilities that we allow to be negative (with the following zeroing-out)
    :param kwargs: dummy
    :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample, represented by diff
    """
    if alpha_as_mean_poster and (alpha is not None):
        poster = 1 - diff * (1 - alpha)
        poster[poster < 0] = 0
        cur_alpha = np.mean(poster)
        if cur_alpha < alpha:
            left_border = alpha
            right_border = 1
        else:
            left_border = 0
            right_border = alpha

            poster_zero = 1 - diff
            poster_zero[poster_zero < 0] = 0
            if np.mean(poster_zero) > alpha:
                left_border = -50
                right_border = 0
                # return 0, poster_zero
        it = 0
        try_alpha = cur_alpha
        while (abs(cur_alpha - alpha) > kwargs.get('tol', 10**-5)) and (it < max_it):
            try_alpha = (left_border + (right_border - left_border) / 2)
            poster = 1 - diff * (1 - try_alpha)
            poster[poster < 0] = 0
            cur_alpha = np.mean(poster)
            if cur_alpha > alpha:
                right_border = try_alpha
            else:
                left_border = try_alpha
            it += 1
        alpha = try_alpha
        if it >= max_it:
            print('Exceeded maximal number of iterations in finding mean_poster=alpha')
    else:
        if alpha is None:
            alpha = 1 - 1 / max(np.quantile(diff, 1 - quantile, interpolation='higher'), 1)
        poster = 1 - diff * (1 - alpha)
        poster[poster < 0] = 0

    return 1 - alpha, poster