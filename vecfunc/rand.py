"""
Generate initial sample for randomized functions.

Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2018 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from numpy.random import *
# https://docs.scipy.org/doc/numpy/reference/routines.random.html
import numpy as np
import scipy.stats
from scipy.stats import truncnorm, lognorm


def trunc_norm(min_val, max_val, mean, std, size=None, increasing=False):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    y = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    if size and increasing:
        y.sort()
    return y


def log_norm(min_val, max_val, mean, std, size=None, increasing=False):
    mu = np.log(mean) - (std ** 2) / 2
    sigma = std

    y = lognorm.rvs(sigma, scale=np.exp(mu), size=size)
    y = np.clip(y, min_val, max_val)
    if size and increasing:
        y.sort()
    return y


def create_dist_beta_mean_max(mean, density=1, max_val=1):
    # E = a / (a+b) => E*(a+b) = a => E*a + E*b = a => E*b = a - E*a => E*b = a*(1-E)
    # => a = b * E/(1-E)
    # => b = a * (1-E)/E
    eps = np.finfo(np.float32).eps
    mean = np.divide(mean, max_val)
    mean = np.clip(mean, eps, 1 - eps)

    beta_a = density
    beta_b = density
    if mean < 0.5:
        beta_a = np.maximum(density * mean / (1 - mean), eps)
    else:
        beta_b = np.maximum(density * (1 - mean) / mean, eps)
    return scipy.stats.beta(beta_a, beta_b, scale=max_val)


def beta_mean_max(mean, density=1, max_val=1, size=None):
    # E = a / (a+b) => E*(a+b) = a => E*a + E*b = a => E*b = a - E*a => E*b = a*(1-E)
    # => a = b * E/(1-E)
    # => b = a * (1-E)/E
    eps = np.finfo(np.float32).eps
    mean = np.divide(mean, max_val)
    mean = np.clip(mean, eps, 1 - eps)
    try:
        size = len(mean)
    except:
        if size is None:
            mean = np.repeat(mean, 1)
        else:
            mean = np.repeat(mean, size)
    u = mean < 0.5
    beta_a = np.repeat(density, size).astype(float)
    beta_b = np.repeat(density, size).astype(float)
    beta_a[u] = np.maximum(density * mean[u] / (1 - mean[u]), eps)
    beta_b[~u] = np.maximum(density * (1 - mean[~u]) / mean[~u], eps)
    ret = scipy.stats.beta.rvs(beta_a, beta_b, size=size, scale=max_val)
    if size is None:
        return ret[0]
    else:
        return ret


def beta_min_max_mean(min_val, max_val, mean, density=1, size=None):
    return beta_mean_max(mean - min_val, density, max_val - min_val, size=size) + min_val


def min_max_avg_dist(min_val, max_val, mean, density=0.1, size=None):
    # p*min + (1-p)*max = mean
    # p*min + max - p*max = mean
    # p*(max-min) = max-mean
    # p = (max-mean)/(max-min)
    p = (max_val-mean) / (max_val-min_val)
    bound_density = 1-density
    p = bound_density*p, bound_density*(1-p), density
    return np.random.choice([min_val, max_val, mean], size=size, p=p)


###############################################################################
# Initial Sample Methods
###############################################################################

def init_sample_nd_uniform(min_val=0., max_val=1., source_limits=None, ndim=2, frequency=None,
                           increasing=True, force_edge=True, local_maximum_count=None):
    if frequency is None:
        frequency = randint(3, 10)
    assert not force_edge or increasing, "Forced edge only applicable in an increasing function."

    if source_limits is None:
        source_limits = [(0., 1.) for _ in range(ndim)]
    source_limits_shape = np.array(source_limits).shape
    assert source_limits_shape == (ndim, 2), "Source limits shape: %s" % (source_limits_shape,)

    d = [np.sort([l, *uniform(l, h, frequency - 2), h]) for l, h in source_limits]
    d.append(uniform(min_val, max_val, frequency))
    if increasing:
        d[-1].sort()
    if force_edge is True:
        d[-1][0] = min_val
        d[-1][-1] = max_val
    elif type(force_edge) in [tuple, list]:
        if force_edge[0]:
            d[-1][0] = min_val
        if force_edge[1]:
            d[-1][-1] = max_val
    if increasing and frequency > 3 and local_maximum_count is not None and local_maximum_count > 0:
        local_maximum_count = min(local_maximum_count, frequency-3)
        replace = np.random.choice(range(1, frequency-2), local_maximum_count, replace=False)
        replace.sort()
        for i in replace:
            d[-1][i], d[-1][i+1] = d[-1][i+1], d[-1][i]

    return tuple(d)


def init_sample_1d_uniform(min_val=0., max_val=1., frequency=None,
                           increasing=True, force_edge=True, local_maximum_count=None):
    return init_sample_nd_uniform(min_val, max_val, None, 1, frequency, increasing, force_edge, local_maximum_count)


def init_sample_1d_gaussian(min_val=0, max_val=1, mean=None, std=None, frequency=None, increasing=False):
    if mean is None:
        mean = uniform(min_val, max_val)
    if std is None:
        std = uniform(np.finfo('float32').eps, (max_val - min_val)/2.)
    if frequency is None:
        frequency = randint(2, 10)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = trunc_norm(min_val, max_val, mean, std, frequency, increasing)
    return x, y


def init_sample_1d_beta(min_val=0, max_val=1, mean=None, density=1, frequency=None, increasing=False):
    if mean is None:
        mean = uniform(min_val, max_val)
    if frequency is None:
        frequency = randint(4, 10)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = beta_mean_max(mean, density, max_val, size=frequency)
    if increasing:
        y.sort()
    return x, y


def init_sample_1d_lognorm(min_val=0, max_val=1, mean=None, var=1, frequency=None, increasing=False):
    if mean is None:
        mean = uniform(min_val, max_val)
    if frequency is None:
        frequency = randint(4, 10)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = log_norm(min_val, max_val, mean, var, size=frequency)
    if increasing:
        y.sort()
    return x, y


def init_sample_1d_beta_multi_mean(min_val=0, max_val=1, mean=np.array([]), density=1,
                                   frequency=None, increasing=False):
    if frequency is None:
        frequency = len(mean)

    if mean is None:
        mean = uniform(min_val, max_val, size=frequency)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = beta_mean_max(mean, density, max_val, size=frequency)
    if increasing:
        y.sort()
    return x, y


def init_sample_1d_norm_multi_mean(min_val=0, max_val=1, mean=np.array([]), var=1, frequency=None, increasing=False):
    if frequency is None:
        frequency = randint(4, 10)
    if mean is None:
        mean = uniform(min_val, max_val, size=frequency)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = np.array([trunc_norm(min_val, max_val, m, var) for m in mean])
    if increasing:
        y.sort()
    return x, y


def init_sample_1d_lognorm_multi_mean(min_val=0, max_val=1, mean=np.array([]), var=1, frequency=None, increasing=False):
    if frequency is None:
        frequency = randint(4, 10)
    if mean is None:
        mean = uniform(min_val, max_val, size=frequency)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = np.array([log_norm(min_val, max_val, m, var) for m in mean])
    if increasing:
        y.sort()
    return x, y


def init_sample_1d_duo_normal(min_val, max_val, normal1, normal2, ratio=0.5,
                              frequency=None, increasing=False):
    if frequency is None:
        frequency = randint(2, 10)

    x = uniform(0, 1, frequency - 2)
    x.sort()
    x = np.array([0., *x, 1.])

    y = []
    for i, (mean, std) in enumerate((normal1, normal2)):
        r = ratio if i == 0 else 1-ratio
        cur_freq = np.round(frequency * r).astype(int)
        a, b = (min_val - mean) / std, (max_val - mean) / std
        y.extend(truncnorm.rvs(a, b, loc=mean, scale=std, size=cur_freq))

    if increasing:
        y.sort()
    else:
        np.random.shuffle(y)

    return x, y


# noinspection PyUnusedLocal
def init_sample_1d_gal(min_val=0, max_val=1, increasing=True):
    u = np.random.uniform(0, 1)
    x = [0, u, 1]
    y = [min_val, (1-u)*max_val, max_val]
    return x, y


def init_sample_1d_concave(min_val=0, max_val=1, x_safe=0.2, y_safe=0.2):
    max_x_val = 1-(x_safe*2)
    ux = beta_mean_max([0.5*max_x_val], 0.7, max_x_val)[0] + x_safe
    max_y_val = 1-ux
    uy = beta_mean_max([0.7*max_y_val], 3, max_y_val)[0] + ux

    # ux = np.random.uniform(0 + x_safe, 1 - x_safe)
    # uy = np.random.uniform(min(ux*(1+y_safe), 0.88), 0.9)
    x = np.array([0, ux, 1])
    y = np.array([min_val, uy, max_val])
    return x, y


def init_sample_nd_exp_grad(dim):
    try:
        dim = tuple(dim)
    except:
        dim = (dim,)
    ndim = len(dim)
    res_size = np.prod(dim)

    d = np.abs(np.random.exponential(1, res_size)).astype('float32').reshape(dim)
    if ndim == 1:
        d[0] = 0
        d = np.cumsum(d)
    elif ndim == 2:
        d[0, 0] = 0
        d[0, :] = np.cumsum(d[0, :])
        d[:, 0] = np.cumsum(d[:, 0])
        for i, j in np.ndindex(*dim):
            d[i, j] += np.max([d[i, j - 1], d[i - 1, j]])
    else:
        raise ValueError("Exponential gradient does not support %d dims" % ndim)

    return d
