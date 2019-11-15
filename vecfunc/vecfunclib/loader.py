"""
Loads the vecfunc module (C++).
Compiles it if necessary.

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
import subprocess
import numpy as np
import os
import sys
import ctypes
import inspect

"""
Read/write requirements from the numpy array.
 - ‘C_CONTIGUOUS’ (‘C’) - ensure a C-contiguous array
 - ‘ALIGNED’ (‘A’)      - ensure a data-type aligned array
 - ‘OWNDATA’ (‘O’)      - ensure an array that owns its own data
 - ‘WRITEABLE’ (‘W’)    - ensure a writable array

See numpy.require documentation for more information.
"""
read_req  = ('C', 'A', 'O')
write_req = (*read_req, 'W')


module_template = "vecfunc_%sd_%s.so"
subpath_options = "vecfunclib/bin", "vecfunclib", "bin", "."


__lib__ = {}
__force_compile__ = False


def force_compile(force=True):
    global __force_compile__
    __force_compile__ = force


def locate_lib_path(fname):
    """ Locate a file in the optional sub-folders"""
    curpath = os.path.dirname(os.path.abspath(__file__))

    while curpath != '/':
        for subpath in subpath_options:
            file_path = os.path.join(curpath, subpath, fname)
            if os.path.isfile(file_path):
                return file_path
        curpath = os.path.normpath(os.path.join(curpath, '..'))

    return None


def normalize_parameters(ndim, dtype):
    """ Return the parameters in a normalized form"""
    if dtype == float:
        dtype = 'float64'
    elif dtype == int:
        dtype = 'int64'

    dtype = dtype.lower().strip()

    return int(ndim), dtype


def locate_dll(ndim, dtype):
    """ Locate the module's DLL file """
    return locate_lib_path(module_template % (ndim, dtype))


def make_lib(ndim, dtype):
    """ Compile the module to specific parameters """
    ndim, dtype = normalize_parameters(ndim, dtype)

    make_file_path = locate_lib_path("makefile")
    cwd = os.path.dirname(make_file_path)

    params_str = "(dim=%s, value=%s)" % (ndim, dtype)
    cmd = "make dim=%s value=%s" % (ndim, dtype)
    caller = inspect.stack()[2]
    print(f"Building vecfunc module for: {params_str}. "
          f"CMD: {cmd}. Called from: {caller[1]} ({caller[2]}). "
          f"In folder: {cwd}.", file=sys.stderr)
    ret = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.stderr or ret.returncode != 0:
        print(str(ret.stderr, 'utf-8'), file=sys.stderr)
        raise RuntimeError("Could not compile module for the parameters: %s." % params_str)


def get_types(ndim, dtype):
    return dict(
        vec_size_t=ctypes.c_uint32 * ndim,
        vecfunc_type=np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, flags=read_req),
        vecfunc_gradient_type=np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim + 1, flags=write_req),
        pts_type=np.ctypeslib.ndpointer(dtype='float64', ndim=2, flags=read_req),
        interp_res_type=np.ctypeslib.ndpointer(dtype='float64', ndim=1, flags=write_req),
        cdfs_arr_type=ctypes.POINTER(ctypes.c_double) * ndim,

        vecfunc_type_1d=np.ctypeslib.ndpointer(dtype=dtype, ndim=1, flags=read_req),
        cdf_type=np.ctypeslib.ndpointer(dtype='float64', ndim=1, flags=read_req),
        ret_cumsum_type=np.ctypeslib.ndpointer(dtype='float64', ndim=1, flags=write_req),
    )


def load_lib(ndim, dtype):
    """ Loads and initialize a module library, or use already loaded module """
    ndim, dtype = normalize_parameters(ndim, dtype)
    key = ndim, dtype
    if key in __lib__:
        return __lib__[key]

    if __force_compile__:
        make_lib(ndim, dtype)

    dll_path = locate_dll(ndim, dtype)
    if dll_path is None:
        make_lib(ndim, dtype)
        dll_path = locate_dll(ndim, dtype)
        if dll_path is None:
            params_str = "(dim=%s, value=%s)" % (ndim, dtype)
            raise RuntimeError("No module was compiled for the parameters: %s." % params_str)
    lib = ctypes.cdll.LoadLibrary(dll_path)

    t = get_types(ndim, dtype)

    # Vecfunc init functions
    lib.is_rising.argtypes = (t['vecfunc_type'], t['vec_size_t'])
    lib.is_rising.restype = ctypes.c_bool

    lib.fix_rising.argtypes = (t['vecfunc_type'], t['vec_size_t'])

    lib.fix_concave_rising.argtypes = (t['vecfunc_type'], t['vec_size_t'])

    lib.calc_gradients.argtypes = (t['vecfunc_type'], t['vec_size_t'], t['vecfunc_gradient_type'], ctypes.c_uint32)

    lib.interp.argtypes = (t['vecfunc_type'], t['vec_size_t'], t['pts_type'], t['interp_res_type'], ctypes.c_uint32)
    lib.interp_triangulate.argtypes = (t['vecfunc_type'], t['vec_size_t'], t['pts_type'], t['interp_res_type'],
                                       ctypes.c_uint32)

    lib.expected_value.argtypes = (t['vecfunc_type'], t['vec_size_t'], t['cdfs_arr_type'], t['vec_size_t'])
    lib.expected_value.restype = ctypes.c_double

    lib.expected_value_1d.argtypes = (t['vecfunc_type_1d'], ctypes.c_uint32, t['cdf_type'], ctypes.c_uint32)
    lib.expected_value_1d.restype = ctypes.c_double

    lib.expected_value_cumsum.argtypes = (t['vecfunc_type_1d'], ctypes.c_uint32, t['cdf_type'], ctypes.c_uint32,
                                          t['ret_cumsum_type'])

    ret = lib, t
    __lib__[key] = ret
    return ret
