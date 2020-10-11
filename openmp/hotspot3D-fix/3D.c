#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <malloc.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#define STR_SIZE (256)
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

typedef double FLOAT;

/* chip parameters	*/
FLOAT t_chip = 0.0005;
FLOAT chip_height = 0.016;
FLOAT chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
FLOAT amb_temp = 80.0;

void readinput(FLOAT *vect, int grid_rows, int grid_cols, int layers,
               char *file) {
  int i, j, k;
  FILE *fp;
  char str[STR_SIZE];
  FLOAT val;

  if ((fp = fopen(file, "r")) == 0)
    printf("The file was not opened");
  fread(vect, sizeof(FLOAT), grid_rows * grid_cols * layers, fp);
  fclose(fp);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, int layers,
                 char *file) {

  int i, j, k, index = 0;
  FILE *fp;
  char str[STR_SIZE];

  if ((fp = fopen(file, "w")) == 0)
    printf("The file was not opened\n");

  for (i = 0; i < grid_rows; i++)
    for (j = 0; j < grid_cols; j++)
      for (k = 0; k < layers; k++) {
        sprintf(str, "%d\t%g\n", index,
                vect[i * grid_cols + j + k * grid_rows * grid_cols]);
        fputs(str, fp);
        index++;
      }

  fclose(fp);
}

void computeTempCPU(FLOAT *pIn, FLOAT *tIn, FLOAT *tOut, int nx, int ny, int nz,
                    FLOAT Cap, FLOAT Rx, FLOAT Ry, FLOAT Rz, FLOAT dt,
                    int numiter) {
  FLOAT ce, cw, cn, cs, ct, cb, cc;
  FLOAT stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;

  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

  int c, w, e, n, s, b, t;
  int x, y, z;
  int i = 0;
  do {
    for (z = 0; z < nz; z++) {
      for (y = 0; y < ny; y++) {
        for (x = 0; x < nx; x++) {
          c = x + y * nx + z * nx * ny;

          w = (x == 0) ? c : c - 1;
          e = (x == nx - 1) ? c : c + 1;
          n = (y == 0) ? c : c - nx;
          s = (y == ny - 1) ? c : c + nx;
          b = (z == 0) ? c : c - nx * ny;
          t = (z == nz - 1) ? c : c + nx * ny;

          tOut[c] = tIn[c] * cc + tIn[n] * cn + tIn[s] * cs + tIn[e] * ce +
                    tIn[w] * cw + tIn[t] * ct + tIn[b] * cb +
                    (dt / Cap) * pIn[c] + ct * amb_temp;
        }
      }
    }
    FLOAT *temp = tIn;
    tIn = tOut;
    tOut = temp;
    i++;
  } while (i < numiter);
}

FLOAT accuracy(FLOAT *arr1, FLOAT *arr2, int len) {
  FLOAT err = 0.0;
  int i;
  for (i = 0; i < len; i++) {
    err += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
  }

  return (FLOAT)sqrt(err / len);
}

void computeTempOMP(FLOAT *restrict pIn, FLOAT *restrict tIn,
                    FLOAT *restrict tOut, uint64_t nx, uint64_t ny, uint64_t nz,
                    FLOAT Cap, FLOAT Rx, FLOAT Ry, FLOAT Rz, FLOAT dt,
                    int numiter) {

  FLOAT ce, cw, cn, cs, ct, cb, cc;

  FLOAT stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;
  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

  {
    int count = 0;
    FLOAT *tIn_t = tIn;
    FLOAT *tOut_t = tOut;

    do {
#ifdef GEM_FORGE
      m5_work_begin(0, 0);
#endif

#pragma omp parallel for schedule(static)                                      \
    firstprivate(tIn_t, pIn, tOut_t, ce, cw, cn, cs, ct, cb, cc, nx, ny, nz,   \
                 amb_temp, dt, Cap)
      for (uint64_t z = 1; z < nz - 1; z++) {
#ifdef GEM_FORGE_FIX_INPUT
        for (uint64_t y = 0; y < 512; y++) {
#pragma omp simd
          for (uint64_t x = 0; x < 512; x++) {
            uint64_t c = x + y * 512 + z * 512 * 512;
            uint64_t idxT = c - 512 * 512;
            uint64_t idxB = c + 512 * 512;
            uint64_t idxN = c - 512;
            uint64_t idxS = c + 512;
#else
        for (uint64_t y = 0; y < ny; y++) {
#pragma omp simd
          for (uint64_t x = 0; x < nx; x++) {
            uint64_t c = x + y * nx + z * nx * ny;
            uint64_t idxT = c - nx * ny;
            uint64_t idxB = c + nx * ny;
            uint64_t idxN = c - nx;
            uint64_t idxS = c + nx;
#endif
            /**
             * ! This will access one element out of the array,
             * ! but we leave it there to have perfect vectorization.
             */
            uint64_t idxW = c - 1;
            uint64_t idxE = c + 1;
            FLOAT p = pIn[c];
            FLOAT tc = tIn_t[c];
            FLOAT tw = tIn_t[idxW];
            FLOAT te = tIn_t[idxE];
            FLOAT tn = tIn_t[idxN];
            FLOAT ts = tIn_t[idxS];
            FLOAT tb = tIn_t[idxB];
            FLOAT tt = tIn_t[idxT];
            tOut_t[c] = cc * tc + cw * tw + ce * te + cs * ts + cn * tn +
                        cb * tb + ct * tt + (dt / Cap) * p + ct * amb_temp;
          }
        }
      }
#ifdef GEM_FORGE
      m5_work_end(0, 0);
#endif
      FLOAT *t = tIn_t;
      tIn_t = tOut_t;
      tOut_t = t;
      count++;
    } while (count < numiter);
  }
  return;
}

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <rows/cols> <layers> <iterations> <nthreads> <powerFile> "
          "<tempFile> "
          "<outputFile>\n",
          argv[0]);
  fprintf(
      stderr,
      "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr,
          "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<nthreads - number of threads to use\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial "
                  "power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial "
                  "temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 8) {
    usage(argc, argv);
  }

  char *pfile = argv[5];
  char *tfile = argv[6];
  char *ofile = argv[7];
  int numCols = atoi(argv[1]);
  int numRows = atoi(argv[1]);
  int layers = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  int numThreads = atoi(argv[4]);
  if (numThreads + 2 > layers) {
    numThreads = (layers > 2) ? (layers - 2) : 1;
  }

  /* calculating parameters*/

  FLOAT dx = chip_height / numRows;
  FLOAT dy = chip_width / numCols;
  FLOAT dz = t_chip / layers;

  FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  FLOAT Rx = dy / (2.0 * K_SI * t_chip * dx);
  FLOAT Ry = dx / (2.0 * K_SI * t_chip * dy);
  FLOAT Rz = dz / (K_SI * dx * dy);

  // cout << Rx << " " << Ry << " " << Rz << endl;
  FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  FLOAT dt = PRECISION / max_slope;

  int size = numCols * numRows * layers;

  FLOAT *powerIn = (FLOAT *)calloc(size, sizeof(FLOAT));
  FLOAT *tempCopy = (FLOAT *)malloc(size * sizeof(FLOAT));
  FLOAT *tempIn = (FLOAT *)calloc(size, sizeof(FLOAT));
  FLOAT *tempOut = (FLOAT *)calloc(size, sizeof(FLOAT));
  FLOAT *answer = (FLOAT *)calloc(size, sizeof(FLOAT));

  // readinput(powerIn, numRows, numCols, layers, pfile);
  // readinput(tempIn, numRows, numCols, layers, tfile);
  // memcpy(tempCopy, tempIn, size * sizeof(FLOAT));

  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  printf("%d threads running\n", omp_get_num_threads());
#ifdef GEM_FORGE
  // mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
#pragma omp parallel for schedule(static)                                      \
    firstprivate(powerIn, tempIn, tempOut, layers, numRows, numCols)
  for (uint64_t z = 0; z < layers; z++) {
    for (uint64_t y = 0; y < numRows; y++) {
      for (uint64_t x = 0; x < numCols; x += 64 / sizeof(FLOAT)) {
        uint64_t c = x + y * numCols + z * numCols * numRows;
        volatile FLOAT v1 = powerIn[c];
        volatile FLOAT v2 = tempIn[c];
        volatile FLOAT v3 = tempOut[c];
      }
    }
  }
  m5_reset_stats(0, 0);
#endif
#endif
  computeTempOMP(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx,
                 Ry, Rz, dt, iterations);
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
  free(tempIn);
  free(tempOut);
  free(powerIn);
  return 0;
}
