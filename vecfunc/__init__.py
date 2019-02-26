"""
See vecfunclib.VecFunc for documentation.

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
from vecfunc import vecfunclib, visualize, rand, vecinterp
from vecfunc.vecfunclib import loader


def vecfunc(f, require_write=False):
    return vecfunclib.as_vecfunc(f, require_write)


def is_rising(f):
    return vecfunclib.as_vecfunc(f).is_rising()


def fix_rising(f):
    f = vecfunclib.as_vecfunc(f, require_write=True)
    f.fix_rising()
    return f


def fix_concave_rising(f):
    f = vecfunclib.as_vecfunc(f, require_write=True)
    f.fix_concave_rising()
    return f


def calc_gradients(f, grad_interval=1):
    return vecfunclib.as_vecfunc(f).calc_gradients(grad_interval)


def interp(f, pts):
    return vecfunclib.as_vecfunc(f).interp(pts)


def interp_triangulate(f, pts):
    return vecfunclib.as_vecfunc(f).interp_triangulate(pts)


def expected_value(f, cdfs):
    return vecfunclib.as_vecfunc(f).expected_value(cdfs)


def expected_value_multi_vec(fs, cdfs):
    cdfs_arr, pts_per_interval = fs[0].inner_prepare_cdfs(cdfs)
    return [f.inner_expected_value(cdfs_arr, pts_per_interval) for f in fs]


def expected_value_1d(f, cdf):
    return vecfunclib.as_vecfunc(f).expected_value_1d(cdf)


def expected_value_1d_multi_vec(fs, cdf):
    cdf, pts_per_interval = fs[0].inner_prepare_cdf_1d(cdf, validate=False)
    return [f.inner_expected_value_1d(cdf, pts_per_interval) for f in fs]


def expected_value_cumsum(f, cdf):
    return vecfunclib.as_vecfunc(f).expected_value_cumsum(cdf)


def plot(f, *args, **kwargs):
    return vecfunclib.as_vecfunc(f).plot(*args, **kwargs)
