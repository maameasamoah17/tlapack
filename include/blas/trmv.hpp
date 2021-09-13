// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRMV_HH
#define BLAS_TRMV_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Triangular matrix-vector multiply:
 * \[
 *     x = op(A) x,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * x is a vector,
 * and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero.
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $x = A   x$,
 *     - Op::Trans:     $x = A^T x$,
 *     - Op::ConjTrans: $x = A^H x$.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *                      The diagonal elements of A are not referenced.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] n
 *     Number of rows and columns of the matrix A. n >= 0.
 *
 * @param[in] A
 *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A. lda >= max(1, n).
 *
 * @param[in, out] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @ingroup trmv
 */
template< typename TA, typename TX >
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    blas::idx_t n,
    TA const *A, blas::idx_t lda,
    TX       *x, blas::int_t incx )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    // quick return
    if (n == 0)
        return;

    // for row major, swap lower <=> upper and
    // A => A^T; A^T => A; A^H => A & conj
    bool doconj = false;
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        if (trans == Op::NoTrans) {
            trans = Op::Trans;
        }
        else {
            if (trans == Op::ConjTrans) {
                doconj = true;
            }
            trans = Op::NoTrans;
        }
    }

    bool nonunit = (diag == Diag::NonUnit);
    idx_t kx = (incx > 0 ? 0 : (-n + 1)*incx);

    if (trans == Op::NoTrans && ! doconj) {
        // Form x := A*x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (idx_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero, for consistent NAN handling
                    TX tmp = x[j];
                    for (idx_t i = 0; i < j; ++i) {
                        x[i] += tmp * A(i, j);
                    }
                    if (nonunit) {
                        x[j] *= A(j, j);
                    }
                }
            }
            else {
                // non-unit stride
                idx_t jx = kx;
                for (idx_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    TX tmp = x[jx];
                    idx_t ix = kx;
                    for (idx_t i = 0; i < j; ++i) {
                        x[ix] += tmp * A(i, j);
                        ix += incx;
                    }
                    if (nonunit) {
                        x[jx] *= A(j, j);
                    }
                    jx += incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    // note: NOT skipping if x[j] is zero ...
                    TX tmp = x[j];
                    for (idx_t i = n-1; i >= j+1; --i) {
                        x[i] += tmp * A(i, j);
                    }
                    if (nonunit) {
                        x[j] *= A(j, j);
                    }
                }
            }
            else {
                // non-unit stride
                kx += (n - 1)*incx;
                idx_t jx = kx;
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    // note: NOT skipping if x[j] is zero ...
                    TX tmp = x[jx];
                    idx_t ix = kx;
                    for (idx_t i = n-1; i >= j+1; --i) {
                        x[ix] += tmp * A(i, j);
                        ix -= incx;
                    }
                    if (nonunit) {
                        x[jx] *= A(j, j);
                    }
                    jx -= incx;
                }
            }
        }
    }
    else if (trans == Op::NoTrans && doconj) {
        // Form x := A*x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (idx_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero, for consistent NAN handling
                    TX tmp = x[j];
                    for (idx_t i = 0; i < j; ++i) {
                        x[i] += tmp * conj( A(i, j) );
                    }
                    if (nonunit) {
                        x[j] *= conj( A(j, j) );
                    }
                }
            }
            else {
                // non-unit stride
                idx_t jx = kx;
                for (idx_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    TX tmp = x[jx];
                    idx_t ix = kx;
                    for (idx_t i = 0; i < j; ++i) {
                        x[ix] += tmp * conj( A(i, j) );
                        ix += incx;
                    }
                    if (nonunit) {
                        x[jx] *= conj( A(j, j) );
                    }
                    jx += incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    // note: NOT skipping if x[j] is zero ...
                    TX tmp = x[j];
                    for (idx_t i = n-1; i >= j+1; --i) {
                        x[i] += tmp * conj( A(i, j) );
                    }
                    if (nonunit) {
                        x[j] *= conj( A(j, j) );
                    }
                }
            }
            else {
                // non-unit stride
                kx += (n - 1)*incx;
                idx_t jx = kx;
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    // note: NOT skipping if x[j] is zero ...
                    TX tmp = x[jx];
                    idx_t ix = kx;
                    for (idx_t i = n-1; i >= j+1; --i) {
                        x[ix] += tmp * conj( A(i, j) );
                        ix -= incx;
                    }
                    if (nonunit) {
                        x[jx] *= conj( A(j, j) );
                    }
                    jx -= incx;
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // Form  x := A^T * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    TX tmp = x[j];
                    if (nonunit) {
                        tmp *= A(j, j);
                    }
                    for (idx_t i = j - 1; i != idx_t(-1); --i) {
                        tmp += A(i, j) * x[i];
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                idx_t jx = kx + (n - 1)*incx;
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    TX tmp = x[jx];
                    idx_t ix = jx;
                    if (nonunit) {
                        tmp *= A(j, j);
                    }
                    for (idx_t i = j - 1; i != idx_t(-1); --i) {
                        ix -= incx;
                        tmp += A(i, j) * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (idx_t j = 0; j < n; ++j) {
                    TX tmp = x[j];
                    if (nonunit) {
                        tmp *= A(j, j);
                    }
                    for (idx_t i = j + 1; i < n; ++i) {
                        tmp += A(i, j) * x[i];
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                idx_t jx = kx;
                for (idx_t j = 0; j < n; ++j) {
                    TX tmp = x[jx];
                    idx_t ix = jx;
                    if (nonunit) {
                        tmp *= A(j, j);
                    }
                    for (idx_t i = j + 1; i < n; ++i) {
                        ix += incx;
                        tmp += A(i, j) * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
    else {
        // Form x := A^H * x
        // same code as above A^T * x case, except add conj()
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    TX tmp = x[j];
                    if (nonunit) {
                        tmp *= conj( A(j, j) );
                    }
                    for (idx_t i = j - 1; i != idx_t(-1); --i) {
                        tmp += conj( A(i, j) ) * x[i];
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                idx_t jx = kx + (n - 1)*incx;
                for (idx_t j = n-1; j != idx_t(-1); --j) {
                    TX tmp = x[jx];
                    idx_t ix = jx;
                    if (nonunit) {
                        tmp *= conj( A(j, j) );
                    }
                    for (idx_t i = j - 1; i != idx_t(-1); --i) {
                        ix -= incx;
                        tmp += conj( A(i, j) ) * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (idx_t j = 0; j < n; ++j) {
                    TX tmp = x[j];
                    if (nonunit) {
                        tmp *= conj( A(j, j) );
                    }
                    for (idx_t i = j + 1; i < n; ++i) {
                        tmp += conj( A(i, j) ) * x[i];
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                idx_t jx = kx;
                for (idx_t j = 0; j < n; ++j) {
                    TX tmp = x[jx];
                    idx_t ix = jx;
                    if (nonunit) {
                        tmp *= conj( A(j, j) );
                    }
                    for (idx_t i = j + 1; i < n; ++i) {
                        ix += incx;
                        tmp += conj( A(i, j) ) * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRMV_HH
