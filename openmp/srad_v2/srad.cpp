// srad.cpp : Defines the entry point for the console application.
//

//#define OUTPUT

#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#ifdef GEM_FORGE_FIX_INPUT
#define GEM_FORGE_FIX_COLS 2048
#endif
#endif

void random_matrix(float *I, int rows, int cols);

void usage(int argc, char **argv) {
  fprintf(
      stderr,
      "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <no. of threads> <lambda> "
      "<no. of iter>\n",
      argv[0]);
  fprintf(stderr, "\t<rows>   - number of rows\n");
  fprintf(stderr, "\t<cols>    - number of cols\n");
  fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr, "\t<no. of threads>  - no. of threads\n");
  fprintf(stderr, "\t<lambda>   - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
  exit(1);
}

__attribute__((noinline)) void sumROI(float *J, int64_t cols, int64_t r1,
                                      int64_t r2, int64_t c1, int64_t c2,
                                      float *sum, float *sum2) {
  float s = 0;
  float s2 = 0;
  for (int64_t i = r1; i <= r2; i++) {
    for (int64_t j = c1; j <= c2; j++) {
      float tmp = J[i * cols + j];
      s += tmp;
      s2 += tmp * tmp;
    }
  }
  *sum = s;
  *sum2 = s2;
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    usage(argc, argv);
  }
  uint64_t rows = atoi(argv[1]); // number of rows in the domain
  uint64_t cols = atoi(argv[2]); // number of cols in the domain
  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }
  int r1 = atoi(argv[3]);       // y1 position of the speckle
  int r2 = atoi(argv[4]);       // y2 position of the speckle
  int c1 = atoi(argv[5]);       // x1 position of the speckle
  int c2 = atoi(argv[6]);       // x2 position of the speckle
  int nthreads = atoi(argv[7]); // number of threads
  float lambda = atof(argv[8]); // Lambda value
  int niter = atoi(argv[9]);    // number of iterations

  uint64_t size_I = cols * rows;

  /**
   * Store the intermediate results.
   * dN, dS, dW, dE.
   */
  uint64_t size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  float *I = (float *)aligned_alloc(64, size_I * sizeof(float));
  // float *J = (float *)aligned_alloc(64, size_I * sizeof(float));
  // float *c = (float *)aligned_alloc(64, size_I * sizeof(float));
  // float *delta = (float *)aligned_alloc(64, sizeof(float) * size_I * 4);
  // float *deltaN = (float *)aligned_alloc(64, sizeof(float) * size_I);
  // float *deltaS = (float *)aligned_alloc(64, sizeof(float) * size_I);
  // float *deltaE = (float *)aligned_alloc(64, sizeof(float) * size_I);
  // float *deltaW = (float *)aligned_alloc(64, sizeof(float) * size_I);

  // Make delta offset by some pages.
#ifndef OFFSET_BYTES
#define OFFSET_BYTES 0
#endif
  const int OFFSET_ELEMENTS = OFFSET_BYTES / sizeof(float);
  const int PAGE_SIZE = 4096;
  const int CACHE_BLOCK_SIZE = 64;
  const int size = size_I;
  int totalBytes = 6 * size * sizeof(float) + OFFSET_BYTES;
  int numPages = (totalBytes + PAGE_SIZE - 1) / PAGE_SIZE;
  int *idx = (int *)aligned_alloc(CACHE_BLOCK_SIZE, numPages * sizeof(int));
#pragma clang loop vectorize(disable) unroll(disable) interleave(disable)
  for (int i = 0; i < numPages; ++i) {
    idx[i] = i;
  }
#ifdef RANDOMIZE
#pragma clang loop vectorize(disable) unroll(disable) interleave(disable)
  for (int j = numPages - 1; j > 0; --j) {
    int i = (int)(((float)(rand()) / (float)(RAND_MAX)) * j);
    int tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
#endif
  float *Buffer = (float *)aligned_alloc(PAGE_SIZE, numPages * PAGE_SIZE);
  float *J = Buffer + 0;
  float *c = J + size_I;
  float *deltaN = c + size_I + OFFSET_ELEMENTS;
  float *deltaS = deltaN + size_I;
  float *deltaE = deltaS + size_I;
  float *deltaW = deltaE + size_I;

  // Now we touch all the pages according to the index.
  int elementsPerPage = PAGE_SIZE / sizeof(float);
#pragma clang loop vectorize(disable) unroll(disable) interleave(disable)
  for (int i = 0; i < numPages; i++) {
    int pageIdx = idx[i];
    int elementIdx = pageIdx * elementsPerPage;
    volatile float v = Buffer[elementIdx];
  }

#ifdef GEM_FORGE
  m5_stream_nuca_region(J, sizeof(J[0]), size_I);
  m5_stream_nuca_region(c, sizeof(c[0]), size_I);
  m5_stream_nuca_region(deltaN, sizeof(deltaN[0]), size_I);
  m5_stream_nuca_region(deltaS, sizeof(deltaS[0]), size_I);
  m5_stream_nuca_region(deltaE, sizeof(deltaE[0]), size_I);
  m5_stream_nuca_region(deltaW, sizeof(deltaW[0]), size_I);
  m5_stream_nuca_align(J, J, cols);
  m5_stream_nuca_align(c, J, 0);
  m5_stream_nuca_align(deltaN, J, 0);
  m5_stream_nuca_align(deltaS, J, 0);
  m5_stream_nuca_align(deltaE, J, 0);
  m5_stream_nuca_align(deltaW, J, 0);
  m5_stream_nuca_remap();
#endif

  // printf("Randomizing the input matrix\n");
  // random_matrix(I, rows, cols);

  // for (int k = 0; k < size_I; k++) {
  //   J[k] = (float)exp(I[k]);
  // }

  printf("Start the SRAD main loop\n");
  omp_set_num_threads(nthreads);
  kmp_set_stacksize_s(8 * 1024 * 1024);
#ifdef GEM_FORGE
  // mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  /**
   * Warm them up separately to make sure paddr is continuous.
   */
#define WARM_ARRAY(A)                                                          \
  for (int64_t i = 0; i < rows * cols; i += 64 / sizeof(float)) {              \
    volatile float v = A[i];                                                   \
  }
  WARM_ARRAY(J);
  WARM_ARRAY(c);
  WARM_ARRAY(deltaN);
  WARM_ARRAY(deltaS);
  WARM_ARRAY(deltaW);
  WARM_ARRAY(deltaE);
#undef WARM_ARRAY

// Start the threads.
#pragma omp parallel for firstprivate(rows, cols) schedule(static)
  for (uint64_t i = 0; i < nthreads; ++i) {
    volatile float vj = J[i];
  }
  m5_reset_stats(0, 0);
#endif
#endif
  for (int iter = 0; iter < niter; iter++) {
    float sum = 0;
    float sum2 = 0;
    sumROI(J, cols, r1, r2, c1, c2, &sum, &sum2);
    float meanROI = sum / size_R;
    float varROI = (sum2 / size_R) - meanROI * meanROI;
    float q0sqr = varROI / (meanROI * meanROI);

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#pragma omp parallel for firstprivate(rows, cols, q0sqr) schedule(static)
    for (uint64_t i = 1; i < rows - 1; i++) {
#pragma omp simd
#ifdef GEM_FORGE_FIX_INPUT
      for (uint64_t j = 0; j < GEM_FORGE_FIX_COLS; j++) {
        uint64_t k = i * GEM_FORGE_FIX_COLS + j;
        uint64_t kN = k - GEM_FORGE_FIX_COLS;
        uint64_t kS = k + GEM_FORGE_FIX_COLS;
#else
      for (uint64_t j = 0; j < cols; j++) {
        uint64_t k = i * cols + j;
        uint64_t kN = k - cols;
        uint64_t kS = k + cols;
#endif

        /**
         * ! Avoid i+1.
         * Please avoid have the expression i + 1 in the loop body,
         * as it will break our stream pass to analyze the pattern
         * of i.
         */

        /**
         * ! Accessing j - 1 and j + 1.
         * This will access out of array bound, but keeps the loop trip count
         * a multiple of vectorize width.
         * We cound also do this with padding.
         */

        // directional derivates
        float Jc = J[k];
        float dWValue = J[k - 1] - Jc;
        float dEValue = J[k + 1] - Jc;
        float dNValue = J[kN] - Jc;
        float dSValue = J[kS] - Jc;

        float G2 = (dNValue * dNValue + dSValue * dSValue + dWValue * dWValue +
                    dEValue * dEValue) /
                   (Jc * Jc);

        float L = (dNValue + dSValue + dWValue + dEValue) / Jc;

        float num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
        float den = 1.0f + (.25f * L);
        float qsqr = num / (den * den);

        // diffusion coefficent (equ 33)
        den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
        float cValue = 1.0f / (1.0f + den);
        // saturate diffusion coefficent
        // cValue = (cValue < 0.0f) ? 0.0f : ((cValue > 1.0f) ? 1.0f : cValue);
        c[k] = cValue;
        // uint64_t dk = i * cols * 4 + j * 4;
        // delta[dk + 0] = dNValue;
        // delta[dk + 1] = dSValue;
        // delta[dk + 2] = dWValue;
        // delta[dk + 3] = dEValue;
        deltaN[k] = dNValue;
        deltaS[k] = dSValue;
        deltaW[k] = dWValue;
        deltaE[k] = dEValue;
      }
    }
#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

/**
 * We use dynamic schedule to avoid floated streams concentrated in one bank.
 */
#ifdef GEM_FORGE_DYN_SCHEDULE
#pragma omp parallel for firstprivate(rows, cols, lambda)                      \
    schedule(dynamic, GEM_FORGE_DYN_SCHEDULE)
#else
#pragma omp parallel for firstprivate(rows, cols, lambda) schedule(static)
#endif
    for (uint64_t i = 1; i < rows - 1; i++) {
#ifdef GEM_FORGE_FIX_INPUT
#pragma omp simd
      for (uint64_t j = 0; j < GEM_FORGE_FIX_COLS; j++) {
        uint64_t k = i * GEM_FORGE_FIX_COLS + j;
        uint64_t kS = k + GEM_FORGE_FIX_COLS;
#else
      for (uint64_t j = 0; j < cols; j++) {
        uint64_t k = i * cols + j;
        uint64_t kS = k + cols;
#endif
        // ! Accessing j + 1.
        // Out of bound.
        // diffusion coefficent
        float cN = c[k];
        float cS = c[kS];
        float cW = cN;
        float cE = c[k + 1];
        // divergence (equ 58)
        float dNValue = deltaN[k];
        float dSValue = deltaS[k];
        float dWValue = deltaW[k];
        float dEValue = deltaE[k];
        float D = cN * dNValue + cS * dSValue + cW * dWValue + cE * dEValue;

        // image update (equ 61)
        // J[k] = J[k] + 0.25f * lambda * D;
        // ! GemForge
        // Fix lambda to reduce the number of input for the computation.
        J[k] = J[k] + 0.25f * D;
      }
    }
#ifdef GEM_FORGE
    m5_work_end(1, 0);
#endif
  }

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

  free(I);
  // free(J);
  // free(delta);
  // free(deltaN);
  // free(deltaS);
  // free(deltaE);
  // free(deltaW);
  // free(c);
  free(Buffer);

  return 0;
}

void random_matrix(float *I, int rows, int cols) {

  srand(7);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float)RAND_MAX;
#ifdef OUTPUT
// printf("%g ", I[i * cols + j]);
#endif
    }
#ifdef OUTPUT
// printf("\n");
#endif
  }
}
