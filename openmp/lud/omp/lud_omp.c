#include <omp.h>
#include <stdio.h>

extern int omp_num_threads;

#define BS 16

#define AA(_i, _j) a[offset * size + (_i)*size + (_j) + offset]
#define BB(_i, _j) a[(_i)*size + (_j)]

void lud_diagonal_omp(float *a, int size, int offset) {
  int i, j, k;
  for (i = 0; i < BS; i++) {

    for (j = i; j < BS; j++) {
      for (k = 0; k < i; k++) {
        AA(i, j) = AA(i, j) - AA(i, k) * AA(k, j);
      }
    }

    float temp = 1.f / AA(i, i);
    for (j = i + 1; j < BS; j++) {
      for (k = 0; k < i; k++) {
        AA(j, i) = AA(j, i) - AA(j, k) * AA(k, i);
      }
      AA(j, i) = AA(j, i) * temp;
    }
  }
}

// implements block LU factorization
void lud_omp(float *a, int size) {

  printf("running OMP on host\n");
  omp_set_num_threads(omp_num_threads);
  for (int offset = 0; offset < size - BS; offset += BS) {

    // lu factorization of left-top corner block diagonal matrix
    lud_diagonal_omp(a, size, offset);

    int size_inter = size - offset - BS;
    int chunks_in_inter_row = size_inter / BS;

// calculate perimeter block matrices
//
#pragma omp parallel for default(none)                                         \
    shared(size, chunks_per_inter, chunks_in_inter_row, offset, a)
    for (int chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++) {

      int i_global = offset;
      int j_global = offset;

      // processing top row of blocks.
      j_global += BS * (chunk_idx + 1);
      for (int j = 0; j < BS; j++) {
        for (int i = 0; i < BS; i++) {
          float sum = 0.f;
          for (int k = 0; k < i; k++) {
            float diag = BB(offset + i, offset + k);
            float here = BB(i_global + k, j_global + j);
            sum += diag * here;
          }
          int i_here = i_global + i;
          int j_here = j_global + j;
          float here = BB(i_here, j_here);
          BB(i_here, j_here) = here - sum;
        }
      }

      // processing left column of blocks.
      //
      j_global = offset;
      i_global += BS * (chunk_idx + 1);
      for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
          float sum = 0.f;
          for (int k = 0; k < j; k++) {
            float diag = BB(offset + k, offset + j);
            float here = BB(i_global + i, j_global + k);
            sum += diag * here;
          }
          int i_here = i_global + i;
          int j_here = j_global + j;
          float here = BB(i_here, j_here);
          BB(i_here, j_here) = here / BB(offset + j, offset + j);
        }
      }
    }

    // update interior block matrices.
    int chunks_per_inter = chunks_in_inter_row * chunks_in_inter_row;

#pragma omp parallel for schedule(auto) default(none)                          \
    shared(size, chunks_per_inter, chunks_in_inter_row, offset, a)
    for (int chunk_idx = 0; chunk_idx < chunks_per_inter; chunk_idx++) {
      float sum[BS] __attribute__((aligned(64))) = {0.f};

      int i_global = offset + BS * (1 + chunk_idx / chunks_in_inter_row);
      int j_global = offset + BS * (1 + chunk_idx % chunks_in_inter_row);

      // Basically Block[][] -= Left[][] * Top[][].
      for (int i = 0; i < BS; i++) {
        for (int k = 0; k < BS; k++) {
#pragma omp simd
          for (int j = 0; j < BS; j++) {
            float left = BB(i_global + i, offset + k);
            float top = BB(offset + k, j_global + j);
            sum[j] += left * top;
          }
        }
#pragma omp simd
        for (int j = 0; j < BS; j++) {
          BB((i + i_global), (j + j_global)) -= sum[j];
          sum[j] = 0.f;
        }
      }
    }
  }

  // The last diagonal block.
  lud_diagonal_omp(a, size, size - BS);
}
