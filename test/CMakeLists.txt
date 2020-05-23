find_package(GTest REQUIRED)

include_directories(${LAPACK_BINARY_DIR}/include)
include_directories(/usr/include/eigen3)

add_executable(FooTest FooTest.cpp)
target_link_libraries(FooTest GTest::GTest GTest::Main lapacke cblas ${BLAS_LIBRARIES})

add_test(AllTestsInFoo FooTest)
