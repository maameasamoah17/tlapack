/// @file laqps_full.hpp
/// @author Racheal Asamoah, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAQPS_FULL_HH
    #define TLAPACK_LAQPS_FULL__HH

    #include "tlapack/base/utils.hpp"
    #include "tlapack/blas/nrm2.hpp"
    #include "tlapack/blas/swap.hpp"
    #include "tlapack/lapack/larf.hpp"
    #include "tlapack/lapack/larfg.hpp"
    #include "tlapack/plugins/stdvector.hpp"

namespace tlapack {

// enum class laqpsStrategy : char { exit_when_find_first_non_trusted };

template <class idx_t, class real_t>
struct laqps_full_opts_t {
    idx_t xb = 11;  ///< Block size for norm recomputation inside the panel
    real_t tol = real_t(1);  ///< Tolerance for finding not trusted columns
                             ///< after the projection occurs
    bool exit_when_find_first_need_to_be_recomputed = true;
};

template <class matrix_t, class vector_idx, class vector_t, class vector2_t>
int laqps_full(size_type<matrix_t>& offset,  // will need to be removed,
                                             // useful for printing purposes
               size_type<matrix_t>& kb,
               matrix_t& A,
               vector_idx& jpvt,
               vector_t& tau,
               vector2_t& current_norm_estimates,
               vector2_t& last_computed_norms,
               const laqps_full_opts_t<size_type<matrix_t>,
                                       real_type<type_t<matrix_t>>>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const real_t eps = ulp<real_t>();
    const real_t tol3z = sqrt(eps);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n);
    const idx_t nb = std::min<idx_t>(kb, k);
    const idx_t xb = std::min<idx_t>(opts.xb, n);

    std::vector<T> auxv_;
    auto auxv = new_matrix(auxv_, nb, 1);

    std::vector<T> F_;
    auto F = new_matrix(F_, n, nb);

    std::vector<T> Gwork_;
    auto Gwork = new_matrix(Gwork_, m, 1);

    std::vector<T> Gworkx_;
    auto Ax = new_matrix(Gworkx_, m, xb);

    std::vector<T> Fx_;
    auto Fx = new_matrix(Fx_, xb, nb);

    std::vector<idx_t> location_recompute(xb);
    std::vector<bool> trusted(n);
    std::vector<bool> need_to_be_recomputed(n);

    // Initializing vector trusted
    for (idx_t j = 0; j < n; ++j)
        trusted[j] = true;

    // Initializing vector trusted
    for (idx_t j = 0; j < n; ++j)
        need_to_be_recomputed[j] = false;

    idx_t i;
    idx_t number_of_recompute = 0;

    for (i = 0; i < nb && number_of_recompute == 0; ++i) {
        // Determine ith pivot column and swap if necessary
        jpvt[i] = i;
        for (idx_t j = i + 1; j < n; j++) {
            if (current_norm_estimates[j] > current_norm_estimates[jpvt[i]])
                jpvt[i] = j;
        }
        auto ai = col(A, i);
        auto bi = col(A, jpvt[i]);
        tlapack::swap(ai, bi);
        auto frow1 = row(F, i);
        auto frow2 = row(F, jpvt[i]);
        tlapack::swap(frow1, frow2);
        std::swap(current_norm_estimates[i], current_norm_estimates[jpvt[i]]);
        std::swap(last_computed_norms[i], last_computed_norms[jpvt[i]]);

        // Apply previous Householder reflectors to column K:
        // A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**H.
        // A2 := A2 - A1 F1^H
        auto A1 = slice(A, pair{i, m}, pair{0, i});
        auto A2 = slice(A, pair{i, m}, pair{i, i + 1});
        auto F1 = slice(F, pair{i, i + 1}, pair{0, i});
        gemm(Op::NoTrans, Op::ConjTrans, -one, A1, F1, one, A2);

        // Generate elementary reflector H(k).
        // Transform A2 into a Householder reflector
        auto v = slice(A, pair{i, m}, i);
        larfg(forward, columnwise_storage, v, tau[i]);
        T Aii = A(i, i);
        A(i, i) = one;

        // Compute Kth column of F:
        // Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**H*A(RK:M,K).
        // F2 := tau_i A3^H A2
        auto A3 = slice(A, pair{i, m}, pair{i + 1, n});
        auto F2 = slice(F, pair{i + 1, n}, pair{i, i + 1});
        gemm(Op::ConjTrans, Op::NoTrans, tau[i], A3, A2, F2);

        // Padding F(1:K,K) with zeros.
        for (idx_t j = 0; j <= i; j++) {
            F(j, i) = zero;
        }

        // Incremental updating of F: F(1:N,K) := F(1:N,K) - tau(K) *
        // F(1:N,1:K-1) * A(RK:M,1:K-1)**H * A(RK:M,K)

        // F4 := F4 - tau_i F3 A1^H A2
        // auxv1 := -tau_i A1^H A2
        auto auxv1 = slice(auxv, pair{0, i}, pair{0, 1});
        gemm(Op::ConjTrans, Op::NoTrans, -tau[i], A1, A2, auxv1);
        // F4 := F4 + F3 auxv1
        auto F3 = slice(F, pair{0, n}, pair{0, i});
        auto F4 = slice(F, pair{0, n}, pair{i, i + 1});
        gemm(Op::NoTrans, Op::NoTrans, one, F3, auxv1, one, F4);

        // Update the current row of A: A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K) *
        // F(K+1:N,1:K)**H
        // A5 := A5 - A4 F5^H
        auto A4 = slice(A, pair{i, i + 1}, pair{0, i + 1});
        auto A5 = slice(A, pair{i, i + 1}, pair{i + 1, n});
        auto F5 = slice(F, pair{i + 1, n}, pair{0, i + 1});
        gemm(Op::NoTrans, Op::ConjTrans, -one, A4, F5, one, A5);

        // trusted / non-trusted
        for (idx_t j = i + 1; j < n; j++) {
            std::cout << "Current norm of column " << j << ": "
                      << current_norm_estimates[j] << std::endl;
            if (current_norm_estimates[j] != zero) {
                //                  NOTE: The following 4 lines follow from
                //                  the analysis in Lapack Working Note 176.
                if (trusted[j]) {
                    real_t temp, temp2;

                    temp = tlapack::abs(A(i, j)) / current_norm_estimates[j];
                    temp = max(zero, (one + temp) * (one - temp));
                    temp2 = current_norm_estimates[j] / last_computed_norms[j];
                    temp2 = temp * (temp2 * temp2);
                    trusted[j] = (temp2 > tol3z);
                    if (!trusted[j]) {
                        std::cout << "** at step i = " << offset + i
                                  << ", column j = " << offset + j
                                  << " is not trusted\n";
                    }
                    else {
                        current_norm_estimates[j] =
                            current_norm_estimates[j] * sqrt(temp);
                    }
                }
            }
        }

        // find the max trusted column norm
        real_t max_trusted_current_estimate = zero;
        for (idx_t j = i + 1; j < n; j++) {
            if ((trusted[j]) &&
                (current_norm_estimates[j] > max_trusted_current_estimate))
                max_trusted_current_estimate = current_norm_estimates[j];
        }

        std::cout << "Max trusted at iter " << i << ": "
                  << max_trusted_current_estimate << std::endl;

        // change diffi
        for (idx_t j = i + 1; j < n; j++) {
            if (current_norm_estimates[j] != zero) {
                // if (!trusted[j] &&
                //     max_trusted_current_estimate < current_norm_estimates[j])
                //     { need_to_be_recomputed[j] = true; std::cout << "** at
                //     step i = " << offset + i
                //               << ", column j = " << offset + j
                //               << " is recomputed\n";
                //     number_of_recompute++;
                // if (!trusted[j]){
                if (!trusted[j] && ((0 * max_trusted_current_estimate) <
                                    current_norm_estimates[j])) {
                    need_to_be_recomputed[j] = true;
                    std::cout << "** at step i = " << offset + i
                              << ", column j = " << offset + j
                              << " is to be recomputed\n";
                    number_of_recompute++;
                }
                // else {
                //     real_t temp;

                //     temp = tlapack::abs(A(i, j)) / current_norm_estimates[j];
                //     temp = max(zero, (one + temp) * (one - temp));

                //     current_norm_estimates[j] =
                //         current_norm_estimates[j] * sqrt(temp);
                // }
            }
        }

        //      make decision on whoi will be ecomputed

        if ((number_of_recompute > 0) &&
            (!opts.exit_when_find_first_need_to_be_recomputed)) {
            idx_t i_number_of_recompute = 0;
            for (idx_t j = i + 1; j < n; j++) {
                if (need_to_be_recomputed[j]) {
                    location_recompute[i_number_of_recompute] = j;
                    i_number_of_recompute++;
                }
                if ((i_number_of_recompute == xb) ||
                    ((i_number_of_recompute > 0) && (j == n - 1))) {
                    for (idx_t ixb = 0; ixb < i_number_of_recompute; ixb++) {
                        idx_t jj = location_recompute[ixb];
                        std::cout << "** at step i = " << offset + i
                                  << ", column j = " << offset + jj
                                  << " is ** ** recomputed\n";
                        auto tilA = slice(A, pair{i + 1, m}, pair{jj, jj + 1});
                        auto tilAx =
                            slice(Ax, pair{i + 1, m}, pair{ixb, ixb + 1});
                        lacpy(Uplo::General, tilA, tilAx);
                        auto tilF = slice(F, pair{jj, jj + 1}, pair{0, i + 1});
                        auto tilFx =
                            slice(Fx, pair{ixb, ixb + 1}, pair{0, i + 1});
                        lacpy(Uplo::General, tilF, tilFx);
                    }

                    auto V = slice(A, pair{i + 1, m}, pair{0, i + 1});
                    auto tilAx = slice(Ax, pair{i + 1, m},
                                       pair{0, i_number_of_recompute});
                    auto tilF = slice(Fx, pair{0, i_number_of_recompute},
                                      pair{0, i + 1});
                    gemm(Op::NoTrans, Op::ConjTrans, -one, V, tilF, one, tilAx);

                    for (idx_t ixb = 0; ixb < i_number_of_recompute; ixb++) {
                        idx_t jj = location_recompute[ixb];
                        current_norm_estimates[jj] =
                            nrm2(slice(Ax, pair{i + 1, m}, ixb));
                        last_computed_norms[jj] = current_norm_estimates[jj];
                        trusted[jj] = true;
                        need_to_be_recomputed[jj] = false;
                    }

                    i_number_of_recompute = 0;
                }
            }
            number_of_recompute = 0;
        }

        // Put A(i,i) back to its place
        A(i, i) = Aii;
    }

    kb = i;
    // Apply the block reflector to the rest of the matrix:
    // A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
    // A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)**H.
    auto tilA = slice(A, pair{kb, m}, pair{kb, n});
    auto V = slice(A, pair{kb, m}, pair{0, kb});
    auto tilF = slice(F, pair{kb, n}, pair{0, kb});
    gemm(Op::NoTrans, Op::ConjTrans, -one, V, tilF, one, tilA);

    if (opts.exit_when_find_first_need_to_be_recomputed &&
        number_of_recompute > 0) {
        for (idx_t j = kb; j < n; j++) {
            if (current_norm_estimates[j] != zero) {
                if (need_to_be_recomputed[j]) {
                    // std::cout << "r ****** i = " << kb - 1 << "****** j = "
                    // <<
                    // j
                    //           << "**" << A(kb - 1, j) << " -- "
                    //           << current_norm_estimates[j] << "\n";
                    current_norm_estimates[j] = nrm2(slice(A, pair{kb, m}, j));
                    last_computed_norms[j] = current_norm_estimates[j];
                }
                else {
                }
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LAQPS_TRICKXX_HH