/*
 * Vectorized function.
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
#ifndef VECFUNC_VECFUNC_HPP
#define VECFUNC_VECFUNC_HPP

#include <cstdint>

#include "mat.hpp"


template<typename T, unsigned int D>
class VecFunc : public mat<T,D> {
public:
    using index = typename mat<uint32_t , D>::index;
    using gradient = typename mat<T, D>::gradient;

public:
    template<typename SIZE_TYPE>
    VecFunc(T* m, const SIZE_TYPE& size) : mat<T,D>(m, size) {}
    VecFunc() : mat<T,D>() {}

public:
    bool is_rising() const {
        index i;
        FOR_EACH_MAT_INDEX(*this, i) {
            auto val = (*this)[i];

            FOR_EACH_DIM(d) {
                if (i[d] == 0)
                    continue;
                i[d]--;
                auto down_val = (*this)[i];
                if (val < down_val)
                    return false;
                i[d]++;
            }
        }

        return true;
    }

    void fix_rising() {
        index i;
        FOR_EACH_MAT_INDEX(*this, i) {
            auto& val = (*this)[i];

            FOR_EACH_DIM(d) {
                if (i[d] == 0)
                    continue;
                i[d]--;
                auto down_val = (*this)[i];
                val = val < down_val ? down_val : val;
                i[d]++;
            }
        }
    }

    void fix_concave_rising() {
		index i;
		FOR_EACH_MAT_INDEX(*this, i) {
			auto& val = (*this)[i];

			FOR_EACH_DIM(d) {
				if (i[d] == 0)
					continue;
				i[d]--;
				auto down_val = (*this)[i];
				if (val < down_val)
					val = down_val;
				if (i[d] == 0) {
					i[d]++;
					continue;
				}

				i[d]--;
				auto down_val_2 = (*this)[i];
				i[d] += 2;

				auto low_grad = down_val - down_val_2;
				auto high_grad = val - down_val;
				if (low_grad < high_grad) {
					val = down_val + low_grad;
				}
			}
		}
	}

    void calc_gradients(gradient* res, unsigned int gradInterval=1) {
    	index i;
		FOR_EACH_MAT_INDEX(*this, i) {
			res[this->get_index(i)] = this->calc_gradient(i, gradInterval);
		}
    }

    void interp(const vec<D,double>* pts, double* res, uint32_t n) {
        // Create a vector of
        vec<D, double> I1;
        I1.setall(1);

        vec<D, unsigned int> t_size;
        t_size.setall(2);

        vec<D, double> t[2];
        mat<double, D> t_mat(NULL, t_size);

        for (unsigned int i=0; i<n; i++) {
            const auto& p = pts[i];

            // Get lower sample point
            vec<D, unsigned int> l(p);

            // Calculate T
            FOR_EACH_DIM(d)
                t[1][d] = p[d] - l[d];


            // Fix T's corner cases
            FOR_EACH_DIM(d) {
                if(p[d] < 0) {
                    l[d] = 0;
                    t[1][d] = 0;
                } else if (!(p[d] < this->size[d]-1)) {
                    l[d] = this->size[d]-2;
                    t[1][d] = 1;
                }
            }

            // Calculate T's complement
            FOR_EACH_DIM(d)
                t[0][d] = 1 - t[1][d];

            index a, m;
            double s = 0;
            FOR_EACH_MAT_INDEX(t_mat, a) {
                vec_add(l, a, m);
                double v = (*this)[m];
                FOR_EACH_DIM(d)
                    v *= t[a[d]][d];
                s += v;
            }
            res[i] = s;
        }
    }

    double expected_value(double** cdfs, const index& cdfs_pts_per_interval) {
        // Prepare coefficients
        std::unique_ptr<double[]> coeff[D];
        FOR_EACH_DIM(d) {
            auto dim_size = this->size[d];
            auto s = cdfs_pts_per_interval[d];
            coeff[d].reset(new double[dim_size]);

            for (unsigned int i=0; i<dim_size-1; i++) {
                coeff[d][i] = 0;
                unsigned int p_ind = (s-1) * i;
                for (unsigned int c=1; c<s; c++)
                    coeff[d][i] += cdfs[d][p_ind + c];
            }
            coeff[d][dim_size-1] = s - cdfs[d][(s-1) * (dim_size-1)];

            coeff[d][0] += cdfs[d][0];
            for (unsigned int i=1; i<dim_size; i++) {
                unsigned int p_ind = (s-1) * i;
                for (unsigned int c=1; c<s; c++)
                    coeff[d][i] -= cdfs[d][p_ind - c];
            }
        }

        // Calculate factor
        double factor = 1;
        FOR_EACH_DIM(d)
            factor *= cdfs_pts_per_interval[d];

        // Sum values
        index i;
        double e = 0;
        FOR_EACH_MAT_INDEX(*this, i) {
            double v = (*this)[i];
            FOR_EACH_DIM(d)
                v *= coeff[d][i[d]];
            e += v;
        }

        return e / factor;
    }

private:
	std::unique_ptr<std::unique_ptr<double[]>[]> get_coeff(double** cdfs,
			const index& cdfs_pts_per_interval) {
		auto coeff = std::unique_ptr<std::unique_ptr<double[]>[]>(new std::unique_ptr<double[]>[D]);
		FOR_EACH_DIM(d)
		{
			auto dim_size = this->size[d];
			auto s = cdfs_pts_per_interval[d];
			coeff[d].reset(new double[dim_size]);

			for (unsigned int i = 0; i < dim_size - 1; i++) {
				coeff[d][i] = 0;
				unsigned int p_ind = (s - 1) * i;
				for (unsigned int c = 1; c < s; c++)
					coeff[d][i] += cdfs[d][p_ind + c];
			}
			coeff[d][dim_size - 1] = s - cdfs[d][(s - 1) * (dim_size - 1)];

			coeff[d][0] += cdfs[d][0];
			for (unsigned int i = 1; i < dim_size; i++) {
				unsigned int p_ind = (s - 1) * i;
				for (unsigned int c = 1; c < s; c++)
					coeff[d][i] -= cdfs[d][p_ind - c];
			}
		}

		return coeff;
	}
};


template<typename T>
void vecfunc_interp_triangulate(const VecFunc<T,2>& f, const vec<2,double>* pts, double* res, uint32_t n) {
    // Create a vector of
    vec<2, double> I1;
    I1.setall(1);

    for (unsigned int i=0; i<n; i++) {
        const auto& p = pts[i];

        // Get lower sample point
        vec<2, unsigned int> l(p);
        vec<2, unsigned int> h,m;
        vec<2, double> t;

        // Calculate T
        FOR_EACH_DIM_D(d, 2)
            t[d] = p[d] - l[d];

        // Fix corner cases
        FOR_EACH_DIM_D(d, 2) {
            if(p[d] < 0) {
                l[d] = 0;
                t[d] = 0;
            } else if (!(p[d] < f.size[d]-1)) {
                l[d] = f.size[d]-2;
                t[d] = 1;
            }
        }

        vec_add(l, I1, h);
        vec_add(l, I1, m);
        if (t[0] < t[1])
            m[0] -= 1;
        else
            m[1] -= 1;

        /*
        Barycentric Coordinates

        det = (y2-y3)(x1-x3) + (x3-x2)(y1-y3)

              (y2-y3)(x-x3) + (x3-x2)(y-y3)
        l1 = -------------------------------
                           det

              (y3-y1)(x-x3) + (x1-x3)(y-y3)
        l2 = -------------------------------
                           det

        l3 = 1 - l1 - l2
        */

        vec<3, double> x, y, v;
        x[0] = l[0];
        y[0] = l[1];
        v[0] = f[l];

        x[1] = h[0];
        y[1] = h[1];
        v[1] = f[h];

        x[2] = m[0];
        y[2] = m[1];
        v[2] = f[m];

        double det = (y[1]-y[2])*(x[0]-x[2]) + (x[2]-x[1])*(y[0]-y[2]);

        auto l1 = ( (y[1]-y[2])*(p[0]-x[2]) + (x[2]-x[1])*(p[1]-y[2]) ) / det;
        auto l2 = ( (y[2]-y[0])*(p[0]-x[2]) + (x[0]-x[2])*(p[1]-y[2]) ) / det;
        auto l3 = 1 - l1 - l2;

        res[i] = l1*v[0] + l2*v[1] + l3*v[2];
    }
}


template<typename T>
void vecfunc_expected_value_cumsum(const T* f, uint32_t size, double* cdf, uint32_t cdf_pts_per_interval,
                                   double* res) {
    // Prepare coefficients
    std::unique_ptr<double[]> coeff;
    std::unique_ptr<double[]> survival_coeff;
    auto s = cdf_pts_per_interval;
    double factor = cdf_pts_per_interval;

    coeff.reset(new double[size]);
    survival_coeff.reset(new double[size]);

    for (unsigned int i=0; i<size-1; i++) {
        coeff[i] = 0;
        unsigned int p_ind = (s-1) * i;
        for (unsigned int c=1; c<s; c++)
            coeff[i] += cdf[p_ind + c] / factor;

        survival_coeff[i] = 1 - (cdf[p_ind] / factor);
    }
    coeff[size-1] = 0;
    survival_coeff[size-1] = 1 - (cdf[(s-1) * (size-1)] / factor);

    coeff[0] += cdf[0];
    survival_coeff[0] = 1;
    for (unsigned int i=1; i<size; i++) {
        unsigned int p_ind = (s-1) * i;
        for (unsigned int c=1; c<s; c++) {
            coeff[i] -= cdf[p_ind - c] / factor;
            survival_coeff[i] -= cdf[p_ind - c] / factor;
        }
    }

    // Sum values
    double cur_sum = 0;
    for (unsigned int i=0; i<size; i++) {
        double v = f[i];
        res[i] = cur_sum + v*survival_coeff[i];
        cur_sum += v*coeff[i];
    }
}


template<typename T>
double vecfunc_expected_value_1d(const T* f, uint32_t size, double* cdf, uint32_t cdf_pts_per_interval) {
    // Prepare coefficients
    double coeff;
    auto s = cdf_pts_per_interval;

    // Sum values
    double e = 0;

    coeff = cdf[0];
    for (unsigned int c=1; c<s; c++)
        coeff += cdf[c];

    e += f[0] * coeff;

    for (unsigned int i=1; i<size-1; i++) {
        coeff = 0;
        unsigned int p_ind = (s-1) * i;
        for (unsigned int c=1; c<s; c++) {
            coeff += cdf[p_ind + c];
            coeff -= cdf[p_ind - c];
        }

        e += f[i] * coeff;
    }

    unsigned int p_ind = (s-1) * (size-1);
    coeff = s - cdf[p_ind];
    for (unsigned int c=1; c<s; c++)
        coeff -= cdf[p_ind - c];

    e += f[size-1] * coeff;

    return e / (double)cdf_pts_per_interval;
}


/*
 * VecFunc implementation with its own memory allocated.
 * Should be used only for inner testing.
 */
template<typename T, unsigned int D>
class VecFuncTest : public VecFunc<T, D> {
public:
    using index = typename VecFunc<T, D>::index;

public:
    template<typename SIZE_TYPE>
    VecFuncTest(const SIZE_TYPE& size) : VecFunc<T,D>(NULL, size) {
    	auto sz = this->total_size();
		this->m = new T[sz];
    }
    VecFuncTest() : VecFunc<T,D>() {}

    template<typename SIZE_TYPE>
    void reset(const SIZE_TYPE& size) {
    	VecFunc<T,D>::reset(NULL, size);
    	auto sz = this->total_size();
		this->m = new T[sz];
    }

    ~VecFuncTest() {
    	if (this->m)
    		delete[] this->m;
    }
};

#endif //VECFUNC_VECFUNC_HPP
