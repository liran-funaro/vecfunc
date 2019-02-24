/*
 * Matrix/Tensor with basic operations.
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
#ifndef MAT_HPP_
#define MAT_HPP_

#include <cstdint>
#include <numeric>
#include <memory>

#include "vec.hpp"


#define FOR_EACH_MAT_INDEX(__mat__, __index__)	\
	(__index__).reset(); 						\
	for (bool __cont__##__index__=true; 		\
		__cont__##__index__;					\
		__cont__##__index__=(__mat__).index_inc(__index__))


#define MAX_VALUE (std::numeric_limits<T>::max() - 1)


template<typename T, unsigned int D>
class mat {
	static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");

public:
	using index = vec<D, uint32_t>;
	using gradient = vec<D*2, T>;

public:
	index size;
	index index_multiplier;
	T* m;

public:
	template<typename SIZE_TYPE>
	mat(T* m, const SIZE_TYPE& size) :
			size(size), index_multiplier(calc_index_multiplier(size)), m(m) { }
	mat() : m(NULL) {}

	template<typename SIZE_TYPE>
	void reset(T* m, const SIZE_TYPE& size) {
		this->m = m;
		this->size = size;
		this->index_multiplier = calc_index_multiplier(this->size);
	}

private:
    template<typename SIZE_TYPE>
	static index calc_index_multiplier(const SIZE_TYPE& size) {
        uint32_t multiplier = 1;
		index ind;
		FOR_EACH_DIM_REV(i) {
			ind[i] = multiplier;
			multiplier *= size[i];
		}
		return ind;
	}

public:
	inline unsigned int get_index(const index& ind) const {
		return ind.inner(index_multiplier);
	}

	inline bool index_inc(index& ind) const {
		FOR_EACH_DIM_REV(d) {
			ind[d] += 1;
			if (ind[d] < size[d])
				return true;
			else
				ind[d] = 0;
		}

		return false;
	}

	inline bool is_edge(const index& ind) const {
		FOR_EACH_DIM(d) {
			if (ind[d] == 0 || ind[d] == (this->size[d]-1))
				return true;
		}

		return false;
	}

    inline unsigned int total_size() const {
        return size.size();
    }

	inline T operator[](unsigned int ind) const { return m[ind]; }
	inline T& operator[](unsigned int ind) { return m[ind]; }

	inline T operator[](const index& ind) const {
		return m[this->get_index(ind)];
	}

	inline T& operator[](const index& ind) {
		return m[this->get_index(ind)];
	}

	inline gradient calc_gradient(index& ind, unsigned int gradInterval=1) {
		gradient g;
		auto cur_val = (*this)[ind];

		FOR_EACH_DIM(d) {
			// Up
			g[d*2] = 0;
			if (ind[d]+gradInterval < this->size[d]) {
				ind[d] += gradInterval;
				g[d*2] = (*this)[ind] - cur_val;
				ind[d] -= gradInterval;
			}

			// Down
			g[d*2 + 1] = 0;
			if (ind[d] > gradInterval-1) {
				ind[d] -= gradInterval;
				g[d*2 + 1] = cur_val - (*this)[ind];
				ind[d] += gradInterval;
			}
		}

		return g;
	}

	template <typename R = T>
	inline R sum() const {
		unsigned int sz = total_size();
		R ret = 0;
		for (unsigned int i=0; i<sz; i++)
			ret += m[i];
		return ret;
	}
};

#endif /* MAT_HPP_ */
