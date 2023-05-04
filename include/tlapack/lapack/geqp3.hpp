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
template <class real_t, class idx_t = size_t>
struct geqp3_opts_t : public workspace_opts_t<> {
    inline constexpr geqp3_opts_t(const workspace_opts_t<>& opts = {})
        : workspace_opts_t<>(opts){};

    idx_t nb = 32;  ///< Panel default size
    idx_t xb = 13;  ///< Block size for norm recomputation inside the panel

    /**
     * alpha_trust is how much you want to be fluid on your concept of
     * ``trusting``.
     * - never set above 1. (Otherwise you trust things that should not be
     * trusted.)
     * - if you set at 1, . . .
     * - if you set below 1, you try to recompute quantity that are trusted
     * but close from being not trusted
     * - if you set to 0, you never trust the formula, so you always recompute
     * the norm by doing a projection.
     */
    real_t alpha_trust = real_t(1.);

    /**
     * alpha_max is how much you want to be fluid on your concept of ``max``.
     * You have an array of either (1) an upperbound on your norm estimate, or
     * (2) your norm estimate. And if you are close to the max, you might want
     * to recompute.
     * - never set above 1. (Otherwise you have quantities (upper bounds) that
     * are above the maximum trusted column norm, and that is not a good idea.)
     * - if you set at 0, then you recompye
     */
    real_t alpha_max = real_t(1.);

    // Strategy 1 for recomputation:
    bool exit_when_find_first_need_to_be_recomputed =
        false;  ///< Strategy to exit before the block size is achieved

    LAqpsVariant variant = LAqpsVariant::full_opts;  ///< Variant for LAQPS
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
int geqp3(
    matrix_t& A,
    vector_idx& jpvt,
    vector_t& tau,
    const geqp3_opts_t<real_type<type_t<matrix_t>>, size_type<matrix_t>>& opts)
{
    using work_t = vector_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    const real_t eps = ulp<real_t>();
    const real_t tol3z = sqrt(eps);

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

    /// TODO: For weslley
    // auto vector_of_norms = new_vector(work, 2 * n);
    std::vector<real_t> vector_of_norms(2 * n);

    for (idx_t j = 0; j < n; j++) {
        vector_of_norms[j] = nrm2(col(A, j));
        vector_of_norms[n + j] = vector_of_norms[j];
    }

    std::vector<real_t> fluid_trusted(n);

    for (idx_t j = 0; j < n; ++j) {
        // initialize with a number greater than one.
        // 1 / tol3z would make sense
        fluid_trusted[j] = real_t(1. / tol3z);
    }

    for (idx_t i = 0; i < k;) {
        idx_t ib = std::min<idx_t>(opts.nb, k - i);

        auto Akk = slice(A, pair{i, m}, pair{i, n});
        auto jpvtk = slice(jpvt, pair{i, i + ib});
        auto tauk = slice(tau, pair{i, i + ib});
        auto partial_normsk = slice(vector_of_norms, pair{i, n});
        auto exact_normsk = slice(vector_of_norms, pair{n + i, 2 * n});
        auto fluid_trustedk = slice(fluid_trusted, pair{i, n});

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
            laqps_full_opts_t<idx_t, real_t> optsQPS;
            optsQPS.alpha_max = opts.alpha_max;
            optsQPS.alpha_trust = opts.alpha_trust;
            optsQPS.verbose = false;
            optsQPS.exit_when_find_first_need_to_be_recomputed =
                opts.exit_when_find_first_need_to_be_recomputed;
            optsQPS.xb = opts.xb;
            laqps_full(fluid_trustedk, i, ib, Akk, jpvtk, tauk, partial_normsk,
                       exact_normsk, optsQPS);
        }

        // std::cout << "kb = " << ib << std::endl;

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

template <class matrix_t,
          class vector_idx,
          class vector_t,
          class T = type_t<matrix_t>,
          enable_if_t<(!allow_optblas<pair<matrix_t, T>, pair<vector_t, T>>) ||
                          (layout<matrix_t> != Layout::ColMajor),
                      int> = 0>
inline int geqp3(matrix_t& A, vector_idx& jpvt, vector_t& tau)
{
    return geqp3(A, jpvt, tau, {});
}

#ifdef USE_LAPACKPP_WRAPPERS

template <class matrix_t,
          class vector_idx,
          class vector_t,
          class T = type_t<matrix_t>,
          enable_if_t<(allow_optblas<pair<matrix_t, T>, pair<vector_t, T>>)&&(
                          layout<matrix_t> == Layout::ColMajor),
                      int> = 0>
int geqp3(matrix_t& A, vector_idx& jpvt, vector_t& tau)
{
    // Legacy objects
    auto A_ = legacy_matrix(A);
    auto tau_ = legacy_vector(tau);

    // Constants to forward
    const auto& m = A_.m;
    const auto& n = A_.n;

    // Init pivots
    std::vector<int64_t> piv(n);
    for (size_t i = 0; i < n; ++i)
        piv[i] = 0;

    // Run geqp3
    int info = ::lapack::geqp3(m, n, A_.ptr, A_.ldim, piv.data(), tau_.ptr);

    // Store pivots
    using idx_t = type_t<vector_idx>;
    for (idx_t i = 0; i < min(m, n); ++i) {
        jpvt[i] = (idx_t)piv[i] - 1; // -1 because of Fortran indexing

        // Now, use the new conventions for the pivots
        size_t safety_counter = 0;
        while (jpvt[i] < i && safety_counter < i) {
            jpvt[i] = jpvt[jpvt[i]];
            safety_counter++;
        }
    }

    return info;
}

#endif

}  // namespace tlapack

#endif