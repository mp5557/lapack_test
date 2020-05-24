#include <iostream>

#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <cblas.h>
#include <lapacke.h>

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

TEST(LinearAlgebra, LinearEquationSolver) {
  Eigen::Matrix3f A;
  Eigen::Vector3f b;
  A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
  b << 3, 3, 4;

  Eigen::Matrix3f A_ = A;
  Eigen::Vector3f x = b;
  Eigen::Vector3i ipiv;
  int info = LAPACKE_sgesv(LAPACK_COL_MAJOR, 3, 1, A_.data(), 3, ipiv.data(),
                           x.data(), 3);

  EXPECT_EQ(info, 0);
  EXPECT_TRUE((A * x - b).isZero());
}

TEST(LinearAlgebra, SymmetricEigenSolver) {
  Eigen::Matrix2f A;
  A << 1, 2, 2, 3;

  Eigen::Matrix2f vec = A;
  Eigen::Vector2f eig;
  int info =
      LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', 2, vec.data(), 2, eig.data());
  EXPECT_EQ(info, 0);
  EXPECT_TRUE((A * vec - vec * eig.asDiagonal()).isZero());
}

TEST(LinearAlgebra, InverseByLU) {
  Eigen::Matrix3f A;
  A << 1, 2, 1, 2, 1, 0, -1, 1, 2;

  Eigen::Matrix3f r = A;
  Eigen::Vector3i ipiv;
  int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, 3, 3, r.data(), 3, ipiv.data());
  EXPECT_EQ(info, 0);
  info = LAPACKE_sgetri(LAPACK_COL_MAJOR, 3, r.data(), 3, ipiv.data());
  EXPECT_EQ(info, 0);

  EXPECT_TRUE((A * r - Eigen::Matrix3f::Identity()).isZero());
}

TEST(LinearAlgebra, DeterminantByLU) {
  for (int k = 0; k < 100; k++) {
    Eigen::Matrix3f A = Eigen::Matrix3f::Random();

    Eigen::Matrix3f r = A;
    Eigen::Vector3i ipiv;
    int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, 3, 3, r.data(), 3, ipiv.data());
    EXPECT_EQ(info, 0);

    int count = 0;
    for (int i = 0; i < 3; i++)
      if (ipiv(i) != i + 1)
        count++;

    float det = count % 2 == 0 ? 1.f : -1.f;
    for (int i = 0; i < 3; i++)
      det *= r(i, i);

    EXPECT_TRUE(std::abs(A.determinant() - det) < 1e-5f)
        << det << ' ' << A.determinant() << '\n'
        << A;
  }
}

/* TEST(LinearAlgebra, RankRevealByQR) { */
/*   Eigen::Matrix3f A; */
/*   A << 1, 2, 5, 2, 1, 4, 3, 0, 3; */

/*   Eigen::Matrix3f A_ = A; */
/*   Eigen::Vector3f tau; */
/*   Eigen::Vector3i jpvt = Eigen::Vector3i::Zero(); */

/*   int info = LAPACKE_sgeqp3(LAPACK_COL_MAJOR, 3, 3, A_.data(), 3, jpvt.data(), */
/*                             tau.data()); */
/*   EXPECT_EQ(info, 0); */

/*   std::cout << "A_\n" << A_ << "\ntau\n" << tau << "\njpvt\n" << jpvt; */
/* } */
