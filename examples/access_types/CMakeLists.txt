# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

project( accessTypes CXX )

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED )
endif()

# add the example accessTypes
add_executable( example_accessTypes example_accessTypes.cpp )
target_link_libraries( example_accessTypes PRIVATE tlapack )
