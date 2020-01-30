// srad.cpp : Defines the entry point for the console application.
//

//#define OUTPUT

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
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

int main(int argc, char *argv[]) {
  uint64_t rows, cols;
  int niter = 10;
  int r1, r2, c1, c2;
  float lambda;
  int nthreads;

  if (argc == 10) {
    rows = atoi(argv[1]); // number of rows in the domain
    cols = atoi(argv[2]); // number of cols in the domain
    if ((rows % 16 != 0) || (cols % 16 != 0)) {
      fprintf(stderr, "rows and cols must be multiples of 16\n");
      exit(1);
    }
    r1 = atoi(argv[3]);       // y1 position of the speckle
    r2 = atoi(argv[4]);       // y2 position of the speckle
    c1 = atoi(argv[5]);       // x1 position of the speckle
    c2 = atoi(argv[6]);       // x2 position of the speckle
    nthreads = atoi(argv[7]); // number of threads
    lambda = atof(argv[8]);   // Lambda value
    niter = atoi(argv[9]);    // number of iterations
  } else {
    usage(argc, argv);
  }

  uint64_t size_I = cols * rows;
  uint64_t size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  float *I = (float *)malloc(size_I * sizeof(float));
  float *J = (float *)malloc(size_I * sizeof(float));
  float *c = (float *)malloc(size_I * sizeof(float));

  /**
   * Store the intermediate results.
   * dN, dS, dW, dE.
   */
  float *delta = (float *)malloc(sizeof(float) * size_I * 4);
  float *deltaN = (float *)malloc(sizeof(float) * size_I);
  float *deltaS = (float *)malloc(sizeof(float) * size_I);
  float *deltaE = (float *)malloc(sizeof(float) * size_I);
  float *deltaW = (float *)malloc(sizeof(float) * size_I);

  printf("Randomizing the input matrix\n");

  random_matrix(I, rows, cols);

  for (int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }

  printf("Start the SRAD main loop\n");
  omp_set_num_threads(nthreads);

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

  for (int iter = 0; iter < niter; iter++) {
    float sum = 0;
    float sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        float tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    float meanROI = sum / size_R;
    float varROI = (sum2 / size_R) - meanROI * meanROI;
    float q0sqr = varROI / (meanROI * meanROI);

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#pragma omp parallel for firstprivate(rows, cols) schedule(static)
    for (uint64_t i = 1; i < rows - 1; i++) {
#pragma clang loop vectorize(enable)
      for (uint64_t j = 0; j < cols; j++) {

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
        float dNValue = 0.0f;
        float dSValue = 0.0f;
        float dWValue = 0.0f;
        float dEValue = 0.0f;
        uint64_t k = i * cols + j;
        float Jc = J[k];
        dWValue = J[k - 1] - Jc;
        dEValue = J[k + 1] - Jc;
        dNValue = J[k - cols] - Jc;
        dSValue = J[k + cols] - Jc;
        // if (j > 0) {
        //   dWValue = J[k - 1] - Jc;
        // }
        // if (j < cols - 1) {
        //   dEValue = J[k + 1] - Jc;
        // }
        // if (i > 0) {
        //   dNValue = J[k - cols] - Jc;
        // }
        // if (i < rows - 1) {
        //   dSValue = J[k + cols] - Jc;
        // }

        float G2 = (dNValue * dNValue + dSValue * dSValue + dWValue * dWValue +
                    dEValue * dEValue) /
                   (Jc * Jc);

        float L = (dNValue + dSValue + dWValue + dEValue) / Jc;

        float num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
        float den = 1 + (.25 * L);
        float qsqr = num / (den * den);

        // diffusion coefficent (equ 33)
        den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
        float cValue = 1.0 / (1.0 + den);
        // saturate diffusion coefficent
        cValue = (cValue < 0.0f) ? 0.0f : ((cValue > 1.0f) ? 1.0f : cValue);
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

#pragma omp parallel for firstprivate(rows, cols, lambda) schedule(static)
    for (uint64_t i = 0; i < rows - 1; i++) {
      for (uint64_t j = 0; j < cols; j++) {
        /**
         * ! Accessing j + 1.
         * Out of bound.
         */

        // current index
        uint64_t k = i * cols + j;

        // diffusion coefficent
        float cN = c[k];
        // float cS = cN;
        // if (i < rows - 1) {
        //   cS = c[k + cols];
        // }
        float cS = c[k + cols];
        float cW = cN;
        // float cE = cW;
        // if (j < cols - 1) {
        //   cE = c[k + 1];
        // }
        float cE = c[k + 1];

        // divergence (equ 58)
        // uint64_t dk = i * cols * 4 + j * 4;
        // float dNValue = delta[dk + 0];
        // float dSValue = delta[dk + 1];
        // float dWValue = delta[dk + 2];
        // float dEValue = delta[dk + 3];
        float dNValue = deltaN[k];
        float dSValue = deltaS[k];
        float dWValue = deltaW[k];
        float dEValue = deltaE[k];
        float D = cN * dNValue + cS * dSValue + cW * dWValue + cE * dEValue;

        // image update (equ 61)
        J[k] = J[k] + 0.25 * lambda * D;
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

#ifdef OUTPUT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {

      printf("%.5f ", J[i * cols + j]);
    }
    printf("\n");
  }
#endif

  printf("Computation Done\n");

  free(I);
  free(J);
  free(delta);
  free(deltaN);
  free(deltaS);
  free(deltaE);
  free(deltaW);

  free(c);
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
