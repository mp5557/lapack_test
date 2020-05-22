PROJECT(gsl_test Fortran C CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CBLAS ON)
SET(LAPACKE ON)
# include(FortranCInterface)
ADD_SUBDIRECTORY(lapack)

# set(lapack_DIR /home/bughunter/local_lib/lib/cmake/lapack-3.9.0)
# FIND_PACKAGE(lapack REQUIRED config)

# set(lapacke_DIR /home/bughunter/local_lib/lib/cmake/lapacke-3.9.0)
# FIND_PACKAGE(lapacke REQUIRED config)

# set(cblas_DIR /home/bughunter/local_lib/lib/cmake/cblas-3.9.0)
# FIND_PACKAGE(cblas REQUIRED config)


ADD_EXECUTABLE(foo foo.cc lapacke_example_aux.c)
TARGET_LINK_LIBRARIES(foo lapacke cblas)

