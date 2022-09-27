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
#include <immintrin.h>  // Intel intrinsics for SSE/AVX.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {
  auto rows = a.rows;
  auto cols = b.columns;
  auto result = Matrix<float>(rows, cols);
  auto bT = Matrix<float>(b.columns, b.rows);
  for(size_t r = 0; r < bT.rows; r++) {
    for(size_t c = 0; c < bT.columns; c++) {
      bT(r, c) = b(c, r);
    }
  }
  for(size_t r = 0; r < rows; r++) {
    for(size_t c = 0; c < cols; c++) {
      float tmp = 0.0;
      for(size_t i = 0; i < a.columns/8; i++) {
        __m256 vec_a = _mm256_loadu_ps(&a(r, i*8));
        __m256 vec_b = _mm256_loadu_ps(&bT(c, i*8));
        __m256 vec_r = _mm256_mul_ps(vec_a, vec_b);
        float *r = (float *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
      }
      auto rem = b.rows % 8;
      if (rem != 0) {
        __m256i vec_mask = _mm256_set_epi32(7-rem, 6-rem, 5-rem, 4-rem, 3-rem, 2-rem, 1-rem, 0-rem);
        __m256 vec_a = _mm256_maskload_ps(&a(r, a.columns-rem), vec_mask);
        __m256 vec_b = _mm256_maskload_ps(&bT(c, bT.columns-rem), vec_mask);
        __m256 vec_r = _mm256_mul_ps(vec_a, vec_b);
        float *r = (float *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
      }
      result(r, c) = tmp;
    }
  }
  return result;
}
Matrix<double> multiplyMatricesSIMD(Matrix<double> a, Matrix<double> b) {
  auto rows = a.rows;
  auto cols = b.columns;
  auto result = Matrix<double>(rows, cols);
  auto bT = Matrix<double>(b.columns, b.rows);
  for(size_t r = 0; r < bT.rows; r++) {
    for(size_t c = 0; c < bT.columns; c++) {
      bT(r, c) = b(c, r);
    }
  }
  for(size_t r = 0; r < rows; r++) {
    for(size_t c = 0; c < cols; c++) {

      double tmp = 0.0;
      for(size_t i = 0; i < a.columns/4; i++) {
        __m256d vec_a = _mm256_loadu_pd(&a(r, i*4));
        __m256d vec_b = _mm256_loadu_pd(&bT(c, i*4));    
        __m256d vec_r = _mm256_mul_pd(vec_a, vec_b);
        double *r = (double *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3]);
      }
      auto rem = b.rows % 4;
      if (rem != 0) {
        __m256i vec_mask = _mm256_set_epi32(3-rem, 3-rem, 2-rem, 2-rem, 1-rem, 1-rem, 0-rem, 0-rem);
        __m256d vec_a = _mm256_maskload_pd(&a(r, a.columns-rem), vec_mask);
        __m256d vec_b = _mm256_maskload_pd(&bT(c, bT.columns-rem), vec_mask);
        __m256d vec_r = _mm256_mul_pd(vec_a, vec_b);
        double *r = (double *)&vec_r;
        tmp += (r[0] + r[1] + r[2] + r[3]);
      }
      result(r, c) = tmp;
    }
  }
  return result;
}/*************************************/
