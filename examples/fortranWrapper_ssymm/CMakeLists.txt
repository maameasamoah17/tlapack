# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( fortranWrapper_ssymm C CXX Fortran )

# Load <T>LAPACK
if( NOT TARGET tlapack_fortran )
  find_package( tlapack REQUIRED )
endif()

# add the example fortranModule_saxpy
add_executable( example_fortranWrapper_ssymm example_fortranWrapper_ssymm.f90 )
target_link_libraries( example_fortranWrapper_ssymm PRIVATE tlapack_fortran )
