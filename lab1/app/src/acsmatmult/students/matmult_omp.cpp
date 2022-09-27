// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acsmatmult/matmult.h"
#include <omp.h>  // OpenMP support.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/
Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
  int M = a.rows;
  int N = a.columns;
  int K = b.columns; 
  auto C = Matrix<float>(M,K);
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared (a,b,C,M,N,K) 
  {
  #pragma omp for 
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
          for (int k = 0; k < N; ++k) {
              C(i,j) += a(i,k) * b(k,j);
          }
      }
  }
  }
  return C;
}

Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num_threads) {
  int M = a.rows;
  int N = a.columns;
  int K = b.columns; 
  auto C = Matrix<double>(M,K);
  
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared (a,b,C,M,N,K) 
  { 
  #pragma omp for
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
          for (int k = 0; k < N; ++k) {
              C(i,j) += a(i,k) * b(k,j);
          }
      }
  }
  }
  return C;
}
#pragma GCC pop_options
#pragma GCC pop_options
