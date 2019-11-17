// srad.cpp : Defines the entry point for the console application.
//

//#define OUTPUT

#define OPEN
#define ITERATION
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
  int rows, cols, size_I, size_R, niter = 10, iter, k;
  float *I, *J, q0sqr, sum, sum2, tmp, meanROI, varROI;
  float G2, L, num, den, qsqr;
  float *dN, *dS, *dW, *dE;
  int r1, r2, c1, c2;
  float cN, cS, cW, cE;
  float *c, D;
  float lambda;
  int i, j;
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

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  I = (float *)malloc(size_I * sizeof(float));
  J = (float *)malloc(size_I * sizeof(float));
  c = (float *)malloc(size_I * sizeof(float));

  dN = (float *)malloc(sizeof(float) * size_I);
  dS = (float *)malloc(sizeof(float) * size_I);
  dW = (float *)malloc(sizeof(float) * size_I);
  dE = (float *)malloc(sizeof(float) * size_I);

  printf("Randomizing the input matrix\n");

  random_matrix(I, rows, cols);

  for (k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }

  printf("Start the SRAD main loop\n");

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

#ifdef ITERATION
  for (iter = 0; iter < niter; iter++) {
#endif
    sum = 0;
    sum2 = 0;
    for (i = r1; i <= r2; i++) {
      for (j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

#ifdef OPEN
    omp_set_num_threads(nthreads);
#pragma omp parallel for shared(J, dN, dS, dW, dE, c, rows, cols, iN, iS, jW,  \
                                jE) private(i, j, k, Jc, G2, L, num, den,      \
                                            qsqr)
#endif
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {

        k = i * cols + j;
        float Jc = J[k];

        // directional derivates
        float dNValue = 0.0f;
        float dSValue = 0.0f;
        float dWValue = 0.0f;
        float dEValue = 0.0f;
        if (i > 0) {
          dNValue = J[(i - 1) * cols + j] - Jc;
        }
        if (i + 1 < rows) {
          dSValue = J[(i + 1) * cols + j] - Jc;
        }
        if (j > 0) {
          dWValue = J[k - 1] - Jc;
        }
        if (j + 1 < cols) {
          dEValue = J[k + 1] - Jc;
        }

        G2 = (dNValue * dNValue + dSValue * dSValue + dWValue * dWValue +
              dEValue * dEValue) /
             (Jc * Jc);

        L = (dNValue + dSValue + dWValue + dEValue) / Jc;

        num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
        den = 1 + (.25 * L);
        qsqr = num / (den * den);

        // diffusion coefficent (equ 33)
        den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
        c[k] = 1.0 / (1.0 + den);

        // saturate diffusion coefficent
        if (c[k] < 0) {
          c[k] = 0;
        } else if (c[k] > 1) {
          c[k] = 1;
        }

        dN[k] = dNValue;
        dS[k] = dSValue;
        dW[k] = dWValue;
        dE[k] = dEValue;
      }
    }
#ifdef OPEN
    omp_set_num_threads(nthreads);
#pragma omp parallel for shared(J, c, rows, cols,                              \
                                lambda) private(i, j, k, D, cS, cN, cW, cE)
#endif
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {

        // current index
        k = i * cols + j;

        // diffusion coefficent
        cN = c[k];
        cS = c[iS[i] * cols + j];
        cW = c[k];
        cE = c[i * cols + jE[j]];

        // divergence (equ 58)
        D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

        // image update (equ 61)
        J[k] = J[k] + 0.25 * lambda * D;
      }
    }

#ifdef ITERATION
  }
#endif

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
  free(dN);
  free(dS);
  free(dW);
  free(dE);

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
