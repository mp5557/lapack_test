#include <iostream>

#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <cblas.h>

TEST(MatrixVectorArithmetic, AddSub) {
  // cblas_dcopy: copy vector
  // cblas_daxpy: y = ax + y

  Eigen::Matrix2d a;
  a << 1, 2, 3, 4;

  Eigen::Matrix2d b;
  b << 2, 3, 1, 4;

  Eigen::Matrix2d c;
  const int size = 4;
  cblas_dcopy(size, a.data(), 1, c.data(), 1);
  EXPECT_TRUE((a - c).isZero());

  cblas_daxpy(size, 1, b.data(), 1, c.data(), 1);
  EXPECT_TRUE((a + b - c).isZero());

  cblas_dcopy(size, a.data(), 1, c.data(), 1);
  cblas_daxpy(size, -1, b.data(), 1, c.data(), 1);
  EXPECT_TRUE((a - b - c).isZero());
}

TEST(MatrixVectorArithmetic, Scale) {
  // cblas_dscal: scale vector

  Eigen::Matrix2d a;
  a << 1, 2, 3, 4;

  double scale = 2.;

  Eigen::Matrix2d b = scale * a;

  cblas_dscal(4, scale, a.data(), 1);
  EXPECT_TRUE((a - b).isZero());
}

TEST(MatrixVectorArithmetic, Transpose) {
  // be careful with col major
  Eigen::Matrix<double, 2, 3> a;
  for (int i = 0; i < a.size(); i++)
    a(i) = i;

  Eigen::Matrix<double, 3, 2> b;
  const int n = b.cols();
  for (int i = 0; i < b.rows(); i++)
    cblas_dcopy(n, a.data() + i * a.rows(), 1, b.data() + i, b.rows());
  EXPECT_TRUE((a.transpose() - b).isZero());
}

TEST(MatrixVectorArithmetic, MatrixMatrixMul) {
  Eigen::Matrix<double, 2, 3> a;
  a << 1, 2, 3, 4, 5, 6;

  Eigen::Matrix<double, 3, 2> b;
  b << 5, 6, 7, 8, 9, 10;

  Eigen::Matrix2d c;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3, 1., a.data(),
              2, b.data(), 3, 0., c.data(), 2);
  EXPECT_TRUE((a * b - c).isZero());
}

TEST(MatrixVectorArithmetic, MatrixVecMul) {
  Eigen::Matrix<double, 2, 3> a;
  a << 1, 2, 3, 4, 5, 6;

  Eigen::Vector3d x;
  x << 7, 8, 9;

  Eigen::Vector2d y;
  cblas_dgemv(CblasColMajor, CblasNoTrans, 2, 3, 1., a.data(), 2, x.data(), 1,
              0., y.data(), 1);
  EXPECT_TRUE((a * x - y).isZero());
}
