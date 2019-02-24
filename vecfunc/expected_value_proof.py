"""
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
import itertools
import functools
from sympy import symbols, IndexedBase, simplify, collect, expand
import numpy as np


"""
We want to calculate the expected value of a valuation given a CDF for each dim.
We use the trapeze formula on an evenly sampled function:
   E(v(X)) = sum(Pr(i-1x<i)*(v[i]+v[i-1])/2) (1<=i<n)
If the CDF is sampled on a similar points, then:
   E(v(X)) = sum((CDF[i]-CDF[i-1])*(v[i]+v[i-1])/2) (1<=i<n)
We also use the fact that:
   E(v(X,Y)) = E(E(v(X,Y)|Y=y))
So we can first calculate the expected value for each known value of y,
that is for example each row in the valuation matrix, and then calculate
the expected value from the new vector of expected values.
   E(v(X,Y)) = sum((CDFx[j]-CDFx[j-1])*sum((CDFy[i]-CDFy[i-1])*(v[i]+v[i-1])/2)) (1<=i<n, 1<=j<m)
And the same goes for higher dimensions.
This yield a complexity of (sum(N^i) 1<=i<=M) for a valuation of M dims and N
samples per dim => O(N^(M+1)).

Here we prove, via symbolic library, that we can calculate the exact same thing
by summing each of the valuation sample in a pre calculated factor yielding 
an O(N^M) solution.
"""


##############################################################
# ND Proof Implementations
##############################################################

def inner_expected_value(v, cdf, dim, cdf_dim, pts_per_interval):
    p, d, cd, s = cdf, dim, cdf_dim, pts_per_interval
    r = v[0] * p[0]
    for i in range(1, d):
        for c in range(1, s):
            p_ind = (s - 1) * (i - 1) + c
            r += (v[i] * c / s + v[i - 1] * (s - c) / s) * (p[p_ind] - p[p_ind - 1])
    r += v[d - 1] * (1 - p[cd - 1])
    return r


def get_nd_expected_value_over_sample(v, cdfs, dim, cdfs_dim):
    ndim = len(dim)
    assert len(cdfs) == ndim
    for d, cd in zip(dim, cdfs_dim):
        assert (cd - 1) % (d - 1) == 0
    pts_per_interval = [int((cd - 1) / (d - 1)) + 1 for d, cd in zip(dim, cdfs_dim)]

    # Init W
    w = np.zeros(dim, dtype=object)
    for ind in itertools.product(*(range(dim[i]) for i in range(ndim))):
        w[ind] = v[ind]

    # Calculate expected value for each dim
    for d in reversed(range(ndim)):
        nw = np.zeros(dim[:d], dtype=object)
        for ind in itertools.product(*(range(dim[i]) for i in range(d))):
            nw[ind] = inner_expected_value(w[ind], cdfs[d], dim[d], cdfs_dim[d],
                                           pts_per_interval[d])
        w = nw

    return w


def calc_nd_expected_value_coeff(cdfs, dim, pts_per_interval):
    coeff = []
    for i, (p, d, s) in enumerate(zip(cdfs, dim, pts_per_interval)):
        a = [None] * d
        for j in range(0, d - 1):
            p_ind = (s - 1) * j
            a[j] = sum(p[p_ind + c] for c in range(1, s))
        a[d - 1] = s - p[(s - 1) * (d - 1)]

        a[0] += p[0]
        for j in range(1, d):
            p_ind = (s - 1) * j
            a[j] -= sum(p[p_ind - c] for c in range(1, s))
        coeff.append(a)
    coeff_factor = int(np.prod(pts_per_interval))
    return coeff, coeff_factor


def get_nd_expected_value_over_sample_short(v, cdfs, dim, cdfs_dim):
    assert len(cdfs) == len(dim)
    for d, cd in zip(dim, cdfs_dim):
        assert (cd - 1) % (d - 1) == 0
    pts_per_interval = [int((cd - 1) / (d - 1)) + 1 for d, cd in zip(dim, cdfs_dim)]

    coeff, coeff_factor = calc_nd_expected_value_coeff(cdfs, dim, pts_per_interval)

    r = 0
    for ind in itertools.product(*[range(d) for d in dim]):
        r += v[ind] * functools.reduce(lambda x, y: x * y, (c[i] for c, i in zip(coeff, ind)))
    r /= coeff_factor
    return r


def proof(dim=(4,), cdfs_dim=(4,)):
    v, *cdfs = symbols(' '.join(['v', *('p%s' % i for i in range(len(dim)))]), cls=IndexedBase)
    e1 = get_nd_expected_value_over_sample(v, cdfs, dim, cdfs_dim)
    e2 = get_nd_expected_value_over_sample_short(v, cdfs, dim, cdfs_dim)
    return simplify(e1 - e2) == 0


##############################################################
# ND Vector Implementations
##############################################################

def inner_expected_value_vector(v, cdf, dim, cdf_dim, pts_per_interval):
    p, d, cd, s = cdf, dim, cdf_dim, pts_per_interval
    r = v[0] * p[0]
    for i in range(1, d):
        for c in range(1, s):
            p_ind = (s - 1) * (i - 1) + c
            r += (v[i] * c / s + v[i - 1] * (s - c) / s) * (p[p_ind] - p[p_ind - 1])
    r += v[d - 1] * (1 - p[cd - 1])
    return r


def get_nd_expected_value_vector_over_sample(v, cdfs, dim, cdfs_dim):
    ndim = len(dim)
    assert len(cdfs) == ndim
    for d, cd in zip(dim, cdfs_dim):
        assert (cd - 1) % (d - 1) == 0
    pts_per_interval = [int((cd - 1) / (d - 1)) + 1 for d, cd in zip(dim, cdfs_dim)]

    # Init W
    w = np.zeros(dim, dtype=object)
    for ind in itertools.product(*(range(dim[i]) for i in range(ndim))):
        w[ind] = v[ind]

    # Calculate expected value for each dim
    for d in reversed(range(ndim)):
        nw = np.zeros(dim[:d], dtype=object)
        for ind in itertools.product(*(range(dim[i]) for i in range(d))):
            nw[ind] = inner_expected_value_vector(w[ind], cdfs[d], dim[d], cdfs_dim[d],
                                                  pts_per_interval[d])
        w = nw

    return w


def test_expected_value_vector(dim=(4,), cdfs_dim=(7,)):
    v, *cdfs = symbols(' '.join(['v', *('p%s' % i for i in range(len(dim)))]), cls=IndexedBase)
    e = get_nd_expected_value_vector_over_sample(v, cdfs, dim, cdfs_dim).item()
    e = expand(e)
    return collect(e, [v[ind] for ind in itertools.product(*map(range, dim))])


if __name__ == '__main__':
    print("1D 4              :", proof((4,), (4,)))
    print("1D 4     => 13    :", proof((4,), (13,)))
    print("2D 4,4   => 4,4   :", proof((4, 4), (4, 4)))
    print("2D 4,4   => 7,4   :", proof((4, 4), (7, 4)))
    print("3D 4,4,4 => 4,4,4 :", proof((4, 4, 4), (4, 4, 4)))
    print("3D 4,4,4 => 7,13,4:", proof((4, 4, 4), (7, 13, 4)))
