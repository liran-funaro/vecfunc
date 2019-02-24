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
import numpy as np
from vecfunc import loader
from vecfunc import visualize


def as_vecfunc(f, require_write=False):
    """ Coverts an array to vecfunc instance """
    if isinstance(f, VecFunc):
        if require_write:
            f.require_write()
        return f
    else:
        f = np.array(f)
        return VecFunc(f, require_write=require_write)


class VecFunc:
    """
    Interpret an array as a function that for an integer input yields the value of the array
    in these indices, and interpolate between them for non integer values.
    """

    def __init__(self, arr, require_write=False):
        self.is_require_write = require_write
        r = loader.read_req if not require_write else loader.write_req
        self.arr = np.require(arr, dtype=arr.dtype, requirements=r)

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    def min(self):
        return np.min(self.arr)

    def max(self):
        return np.max(self.arr)

    def argmax(self):
        return np.unravel_index(np.argmax(self.arr), self.shape)

    def normalize(self, boundaries=(0, 1)):
        self.arr -= self.min()
        self.arr /= self.max() / (boundaries[1]-boundaries[0])
        self.arr += boundaries[0]

    def __getitem__(self, ind):
        return self.arr[ind]

    def require_write(self):
        if not self.is_require_write:
            self.arr = np.require(self.arr, dtype=self.dtype, requirements=loader.write_req)
            self.is_require_write = True

    @property
    def ctype_arr_size(self):
        _, data = loader.load_lib(self.ndim, self.dtype)
        return data['vec_size_t'](*self.shape)

    @property
    def _lib(self):
        return self.get_lib()

    def get_lib(self):
        lib, _ = loader.load_lib(self.ndim, self.dtype)
        return lib

    def is_rising(self):
        """ Check if a function is monotonically rising """
        return self._lib.is_rising(self.arr, self.ctype_arr_size)

    def fix_rising(self):
        """ Fix a function to be monotonically rising """
        self.require_write()
        self._lib.fix_rising(self.arr, self.ctype_arr_size)

    def fix_concave_rising(self):
        """ Fix a function to be monotonically rising """
        self.require_write()
        self._lib.fix_concave_rising(self.arr, self.ctype_arr_size)

    def calc_gradients(self, grad_interval=1):
        """ Calculate the up and down gradient in each dimension """
        ret = np.empty((*self.shape, self.ndim * 2), dtype=self.dtype, order='C')
        ret = np.require(ret, dtype=self.dtype, requirements=loader.write_req)
        self._lib.calc_gradients(self.arr, self.ctype_arr_size, ret, grad_interval)
        return ret

    def interp(self, pts):
        """ Interpolate """
        n = len(pts)

        pts = np.require(pts, dtype='float64', requirements=loader.read_req)
        ret = np.empty(n, dtype='float64', order='C')
        ret = np.require(ret, dtype='float64', requirements=loader.write_req)

        self._lib.interp(self.arr, self.ctype_arr_size, pts, ret, n)
        return ret

    def interp_triangulate(self, pts):
        """ Interpolate via Triangulation """
        n = len(pts)

        pts = np.require(pts, dtype='float64', requirements=loader.read_req)
        ret = np.empty(n, dtype='float64', order='C')
        ret = np.require(ret, dtype='float64', requirements=loader.write_req)

        self._lib.interp_triangulate(self.arr, self.ctype_arr_size, pts, ret, n)
        return ret

    def __call__(self, *x):
        assert len(x) == self.ndim
        res_shape = np.array(x[0]).shape
        pts = np.reshape(x, (self.ndim, -1)).T
        res = self.interp(pts)
        return res.reshape(res_shape)

    def expected_value(self, cdfs):
        """ Calculate the expected value given a CDFs """
        cdfs_arr, pts_per_interval = self.inner_prepare_cdfs(cdfs)
        return self.inner_expected_value(cdfs_arr, pts_per_interval)

    def expected_value_1d(self, cdf):
        """ Calculate the expected value given a CDFs (for a 1 dimensional vecfunc). """
        cdf, pts_per_interval = self.inner_prepare_cdf_1d(cdf)
        return self.inner_expected_value_1d(cdf, pts_per_interval)

    def inner_expected_value_1d(self, cdf, pts_per_interval):
        return self._lib.expected_value_1d(self.arr, self.shape[0], cdf, pts_per_interval)

    def inner_prepare_cdfs(self, cdfs):
        """ Internal use: convert a CDF sample to a CDF sample which corresponds to the number of samples
         in the vecfunc. """
        assert len(cdfs) == self.ndim, "Must supply a CDF for each dim: %s CDFs != %s" % (len(cdfs), self.ndim)
        cdfs_shape = [len(c) for c in cdfs]
        for i in range(self.ndim):
            assert (cdfs_shape[i] - 1) % (self.shape[i] - 1) == 0, \
                "CDF (%s) length does not match the function shape: %s != %s" % (i, cdfs_shape, self.shape)

        _, data = loader.load_lib(self.ndim, self.dtype)
        cdfs = [np.require(c, dtype='float64', requirements=loader.read_req) for c in cdfs]
        cdfs_arr = data['cdfs_arr_type'](*[np.ctypeslib.as_ctypes(c) for c in cdfs])
        pts_per_interval = data['vec_size_t'](
            *[int((len(cdfs[i]) - 1) / (self.shape[i] - 1)) + 1 for i in range(self.ndim)])

        return cdfs_arr, pts_per_interval

    def inner_prepare_cdf_1d(self, cdf, validate=True):
        """ Internal use: convert a CDF sample to a CDF sample which corresponds to the number of samples
         in the vecfunc (for a 1 dimensional vecfunc). """
        cdf_size = len(cdf)
        if validate:
            assert len(cdf.shape) == 1, "Must supply a CDF single CDF (cumsum only support 1d)"
            assert (cdf_size - 1) % (self.shape[0] - 1) == 0, \
                "CDF length does not match the function shape: %s != %s" % (cdf_size, self.shape)

        cdf = np.require(cdf, dtype='float64', requirements=loader.read_req)
        pts_per_interval = int((cdf_size - 1) / (self.shape[0] - 1)) + 1
        return cdf, pts_per_interval

    def inner_expected_value(self, cdfs_arr, pts_per_interval):
        """ Calculate the expected value given a CDFs """
        return self._lib.expected_value(self.arr, self.ctype_arr_size, cdfs_arr, pts_per_interval)

    def expected_value_cumsum(self, cdf):
        """ Calculate the expected value given a CDFs """
        assert len(cdf.shape) == 1, "Must supply a CDF single CDF (cumsum only support 1d)"
        cdf_size = len(cdf)
        assert (cdf_size - 1) % (self.shape[0] - 1) == 0, \
            "CDF length does not match the function shape: %s != %s" % (cdf_size, self.shape)

        cdf = np.require(cdf, dtype='float64', requirements=loader.read_req)
        pts_per_interval = int((cdf_size - 1) / (self.shape[0] - 1)) + 1

        ret = np.empty_like(self.arr, dtype='float64', order='C')
        ret = np.require(ret, dtype='float64', requirements=loader.write_req)

        self._lib.expected_value_cumsum(self.arr, self.shape[0], cdf, pts_per_interval, ret)
        return ret

    def plot(self, d_vals=None, d_keys=None, val_key=None, **kwargs):
        """ Plots the vecfunc """
        visualize.visualize_vector(self.arr, d_vals=d_vals, d_keys=d_keys, val_key=val_key, **kwargs)
