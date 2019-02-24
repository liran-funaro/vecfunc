"""
Interpolate, sample and resample vecfunc

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
import numpy as np
from scipy import interpolate
from scipy.interpolate import Rbf

import vecfunc

###############################################################################
# Common interpolation methods
###############################################################################


def refine_chaikin_corner_cutting(coords, refinements=5):
    """
    Implementation of Chaikin's Corner Cutting Scheme.

    For every two points, we need to take the lower part and the upper part in
    the line between them using the ratio 1:3:
        LOWER-POINT = P1 * 0.25 + P2 * 0.75
        UPPER-POINT = P1 * 0.75 + P2 * 0.25
    and add them both to the new points list.
    We also need to add the edge points, so the line will not shrink.

    We build two arrays L and R in a certain way that if we will multiply them
    as follows it will yield the new points list.
        NEW-POINTS = L * 0.75 + R * 0.25
    For example, if we have array of 4 points:
        P = 0 1 2 3
    the L and R arrays will be as follows:
        L = 0 0 1 1 2 2 3 3
        R = 0 1 0 2 1 3 2 3
    where each number corresponds to a point.
    """
    coords = np.array(coords)

    for _ in range(refinements):
        left          = coords.repeat(2, axis=0)
        right         = np.empty_like(left)
        right[0]      = left[0]
        right[2::2]   = left[1:-1:2]
        right[1:-1:2] = left[2::2]
        right[-1]     = left[-1]
        coords = left * 0.75 + right * 0.25

    return coords


def refine_chaikin_corner_cutting_xy(*args, refinements=5):
    return refine_chaikin_corner_cutting(np.array(args).T, refinements=refinements).T


def interp_spline(x, y, smooth=0):
    pts_count = len(x)
    degree = 3 if pts_count > 2 else pts_count - 1
    tck = interpolate.splrep(x, y, s=smooth, k=degree)
    return lambda points: interpolate.splev(points, tck, der=1)


def interp_rbf(x, y, method='inverse', smooth=0):
    """
    'multiquadric': sqrt((r/self.epsilon)**2 + 1)
    'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
    'gaussian': exp(-(r/self.epsilon)**2)
    'linear': r
    'cubic': r**3
    'quintic': r**5
    'thin_plate': r**2 * log(r)
    """
    return Rbf(x, y, function=method, smooth=smooth)


def interp_nd_matrix_rbf(mat, method='inverse', smooth=0):
    axes = [np.linspace(0, 1, d) for d in mat.shape]
    mesh = np.meshgrid(*axes, sparse=False, indexing='ij')
    return Rbf(*mesh, mat, function=method, smooth=smooth)


###############################################################################
# Sample methods at specific points
###############################################################################

def sample_nd_matrix_linear(mat, sample_list):
    sample_list = np.array(sample_list)
    if sample_list.ndim > 1:
        ret_pts = sample_list.T
    else:
        ret_pts = np.array([sample_list])

    result = vecfunc.interp(mat, ret_pts)

    if sample_list.ndim > 1:
        return result
    else:
        return result[0]


def sample_nd_matrix_linear_triangulate(mat, sample_list):
    sample_list = np.array(sample_list)
    if sample_list.ndim > 1:
        ret_pts = sample_list.T
    else:
        ret_pts = np.array([sample_list])

    result = vecfunc.interp_triangulate(mat, ret_pts)

    if sample_list.ndim > 1:
        return result
    else:
        return result[0]


###############################################################################
# Re sample methods
###############################################################################

def resample_1d_function(func, sample_count, boundaries=(0, 1)):
    low, high = boundaries
    x = np.linspace(low, high, sample_count)
    return func(x)


def resample_1d_array(arr, sample_count, boundaries=(0, 1)):
    low, high = boundaries
    arr_x = np.linspace(low, high, len(arr))
    x = np.linspace(low, high, sample_count)
    return np.interp(x, arr_x, arr)


def resample_1d_refine_linear(function_parameters, sample_count, boundaries=(0, 1)):
    ret = refine_chaikin_corner_cutting_xy(*function_parameters)
    low, high = boundaries
    x = np.linspace(low, high, sample_count)
    return np.interp(x, ret[0], ret[1])


def resample_nd_matrix_linear_sample(mat, sample_axes, method='normal'):
    ret_mesh = np.meshgrid(*sample_axes, sparse=False, indexing='ij')
    ret_pts = np.reshape(ret_mesh, (mat.ndim, -1)).T
    if method != 'triangulate':
        return vecfunc.interp(mat, ret_pts)
    else:
        return vecfunc.interp_triangulate(mat, ret_pts)


def resample_nd_matrix_linear_sample_prod(mat, sample_axes, method='normal'):
    return resample_nd_matrix_linear_sample(mat, sample_axes, method=method).reshape([len(s) for s in sample_axes])


def resample_nd_matrix_linear(mat, res_dim, method='normal'):
    sample_axes = [np.linspace(0, d-1, s) for d, s in zip(mat.shape, res_dim)]
    return resample_nd_matrix_linear_sample_prod(mat, sample_axes, method=method)


def resample_nd_matrix_rbf(mat, res_dim):
    func = interp_nd_matrix_rbf(mat)
    res_axes = [np.linspace(0, 1, d) for d in res_dim]
    res_mesh = np.meshgrid(*res_axes, sparse=False, indexing='ij')
    ret = func(*res_mesh)
    del func
    return ret
