/// @file geqp3.hpp
/// @author Racheal Asamoah, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQP3_HH
#define TLAPACK_GEQP3_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/laqps_full.hpp"
#include "tlapack/lapack/laqps_trick.hpp"
#include "tlapack/lapack/laqps_trickx.hpp"
#include "tlapack/lapack/laqps_trickxx.hpp"

namespace tlapack {

enum class LAqpsVariant : char { Trick, TrickX, TrickXX, full_opts };

/**
 * Options struct for geqp3
 */
template <class idx_t = size_t>
struct geqp3_opts_t : public workspace_opts_t<> {
    inline constexpr geqp3_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

    idx_t nb = 32;                                 ///< Panel default size
    idx_t xb = 13;                                 ///< Block size for norm recomputation inside the panel
    LAqpsVariant variant = LAqpsVariant::TrickXX;  ///< Variant for LAQPS
};

template <class matrix_t, class vector_idx, class vector_t>
inline constexpr void geqp3_worksize(const matrix_t& A,
                                     const vector_idx& jpvt,
                                     const vector_t& tau,
                                     workinfo_t& workinfo,
                                     const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // constants
    const idx_t n = ncols(A);

    const workinfo_t myWorkinfo(sizeof(real_t), 2 * n);
    workinfo.minMax(myWorkinfo);
}

/** Computes a QR factorization of a matrix A using partial pivoting with
 * blocking.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * TODO: Put information about jpvt.
 *
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *      TODO: Update information.
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_idx, class vector_t>
int geqp3(matrix_t& A,
          vector_idx& jpvt,
          vector_t& tau,
          const geqp3_opts_t<size_type<matrix_t>>& opts = {})
{
    using work_t = vector_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // Functor
    Create<work_t> new_vector;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n);

    // check arguments
    tlapack_check((idx_t)size(tau) >= std::min<idx_t>(m, n));

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        geqp3_worksize(A, jpvt, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();
    auto vector_of_norms = new_vector(work, 2 * n);

    for (idx_t j = 0; j < n; j++) {
        vector_of_norms[j] = nrm2(col(A, j));
        vector_of_norms[n + j] = vector_of_norms[j];
    }

    for (idx_t i = 0; i < k;) {
        idx_t ib = std::min<idx_t>(opts.nb, k - i);

        auto Akk = slice(A, pair{i, m}, pair{i, n});
        auto jpvtk = slice(jpvt, pair{i, i + ib});
        auto tauk = slice(tau, pair{i, i + ib});
        auto partial_normsk = slice(vector_of_norms, pair{i, n});
        auto exact_normsk = slice(vector_of_norms, pair{n + i, 2 * n});

        if (opts.variant == LAqpsVariant::Trick) {
            laqps_trick(ib, Akk, jpvtk, tauk, partial_normsk, exact_normsk,
                        laqps_trick_opts_t<idx_t>{opts.nb});
        }
        else if (opts.variant == LAqpsVariant::TrickX) {
            laqps_trickx(i, ib, Akk, jpvtk, tauk, partial_normsk, exact_normsk,
                         laqps_trickx_opts_t<idx_t>{opts.nb});
        }
        else if (opts.variant == LAqpsVariant::TrickXX) {
            laqps_trickxx(i, ib, Akk, jpvtk, tauk, partial_normsk, exact_normsk,
                          laqps_trickxx_opts_t<idx_t>{opts.nb});
        }
        else if (opts.variant == LAqpsVariant::full_opts) {
            laqps_full(i, ib, Akk, jpvtk, tauk, partial_normsk, exact_normsk,
                       laqps_full_opts_t<idx_t, real_t>{opts.xb,real_t(10),false});
        }

        std::cout << "kb = " << ib << std::endl;

        // Swap the columns above Akk
        auto A0k = slice(A, pair{0, i}, pair{i, n});
        for (idx_t j = 0; j != ib; j++) {
            auto vect1 = tlapack::col(A0k, j);
            auto vect2 = tlapack::col(A0k, jpvtk[j]);
            tlapack::swap(vect1, vect2);
        }

        for (idx_t j = 0; j != ib; j++) {
            jpvtk[j] += i;
        }
        i += ib;
    }
    return 0;
}

}  // namespace tlapack

#endif