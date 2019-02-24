/*
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
#ifndef INCLUDE_VECFUNC_TYPES_HPP_
#define INCLUDE_VECFUNC_TYPES_HPP_

#include <cstdint>
#include <vecfunc.hpp>

#ifndef DIM
#define DIM 2
#endif

#ifndef VALUE
#define VALUE uint32_t
#endif

using int32 = int32_t;
using int64 = int64_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using float32 = float;
using float64 = double;

typedef VecFunc<VALUE,DIM> TDVecFunc;
typedef VecFuncTest<VALUE,DIM> TDVecFuncTest;


#endif /* INCLUDE_VECFUNC_TYPES_HPP_ */
