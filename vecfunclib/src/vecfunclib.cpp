/*
 * VecFunc external library.
 *
 * Author: Liran Funaro <liran.funaro@gmail.com>
 *
 * Copyright (C) 2006-2018 Liran Funaro
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <string>
#include <cstdint>

#include <vecfunc_types.hpp>
#include <vecfunc.hpp>


// Shared library interface
extern "C" {

bool is_rising(VALUE* val, uint32_t* val_size) {
    TDVecFunc v(val, val_size);
    return v.is_rising();
}

void fix_rising(VALUE* val, uint32_t* val_size) {
    TDVecFunc v(val, val_size);
    v.fix_rising();
}

void fix_concave_rising(VALUE* val, uint32_t* val_size) {
    TDVecFunc v(val, val_size);
    v.fix_concave_rising();
}

void calc_gradients(VALUE* val, uint32_t* val_size, VALUE* res, uint32_t grad_interval) {
	TDVecFunc v(val, val_size);
	v.calc_gradients((TDVecFunc::gradient*)res, grad_interval);
}

void interp(VALUE* val, uint32_t* val_size, double* pts, double* res, uint32_t n) {
    TDVecFunc v(val, val_size);
    v.interp((vec<DIM,double>*)pts, res, n);
}

void interp_triangulate(VALUE* val, uint32_t* val_size, double* pts, double* res, uint32_t n) {
    if (DIM != 2)
        return;

    VecFunc<VALUE,2> v(val, val_size);
    vecfunc_interp_triangulate(v, (vec<2,double>*)pts, res, n);
}

double expected_value(VALUE* val, uint32_t* val_size, double** cdfs, uint32_t* cdfs_pts_per_interval) {
    TDVecFunc v(val, val_size);
    TDVecFunc::index vec_cdfs_pts_per_interval(cdfs_pts_per_interval);
    return v.expected_value(cdfs, vec_cdfs_pts_per_interval);
}

double expected_value_1d(VALUE* val, uint32_t size, double* cdf, uint32_t cdf_pts_per_interval) {
    if (DIM != 1)
        return 0;
    return vecfunc_expected_value_1d(val, size, cdf, cdf_pts_per_interval);
}

void expected_value_cumsum(VALUE* val, uint32_t size, double* cdf, uint32_t cdf_pts_per_interval, double* res) {
    if (DIM != 1)
        return;
    vecfunc_expected_value_cumsum(val, size, cdf, cdf_pts_per_interval, res);
}

} // extern "C"
