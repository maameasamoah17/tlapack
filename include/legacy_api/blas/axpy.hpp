// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_AXPY_HH__
#define __TLAPACK_LEGACY_AXPY_HH__

#include "legacy_api/base/utils.hpp"
#include "legacy_api/base/types.hpp"
#include "blas/axpy.hpp"

namespace tlapack {

/**
 * Add scaled vector, $y = \alpha x + y$.
 * 
 * Wrapper to axpy(
    const alpha_t& alpha,
    const vectorX_t& x, vectorY_t& y ).
 *
 * @param[in] n
 *     Number of elements in x and y. n >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, y is not updated.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in,out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup axpy
 */
template< typename TX, typename TY >
void axpy(
    idx_t n,
    scalar_type<TX, TY> alpha,
    TX const *x, int_t incx,
    TY       *y, int_t incy )
{
    tlapack_check_false( incx == 0 );
    tlapack_check_false( incy == 0 );

    // quick return
    if( n <= 0 ) return;
    
    tlapack_expr_with_2vectors(
        _x, TX, n, x, incx,
        _y, TY, n, y, incy,
        return axpy( alpha, _x, _y )
    );
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_LEGACY_AXPY_HH__
