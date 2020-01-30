#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

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

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void fatal(char *s) { fprintf(stderr, "Error: %s\n", s); }

void readinput(float *vect, int grid_rows, int grid_cols, int layers,
               char *file) {
  int i, j, k;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if ((fp = fopen(file, "r")) == 0)
    fatal("The file was not opened");
  fread(vect, sizeof(float), grid_rows * grid_cols * layers, fp);
  fclose(fp);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, int layers,
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

void computeTempCPU(float *pIn, float *tIn, float *tOut, int nx, int ny, int nz,
                    float Cap, float Rx, float Ry, float Rz, float dt,
                    int numiter) {
  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
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
    float *temp = tIn;
    tIn = tOut;
    tOut = temp;
    i++;
  } while (i < numiter);
}

float accuracy(float *arr1, float *arr2, int len) {
  float err = 0.0;
  int i;
  for (i = 0; i < len; i++) {
    err += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
  }

  return (float)sqrt(err / len);
}
void computeTempOMP(float *pIn, float *tIn, float *tOut, uint64_t nx,
                    uint64_t ny, uint64_t nz, float Cap, float Rx, float Ry,
                    float Rz, float dt, int numiter) {

  float ce, cw, cn, cs, ct, cb, cc;

  float stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;

  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

  {
    int count = 0;
    float *tIn_t = tIn;
    float *tOut_t = tOut;

    do {

#ifdef GEM_FORGE
      m5_work_begin(0, 0);
#endif

#pragma omp parallel for schedule(static)                                      \
    firstprivate(tIn_t, pIn, tOut_t, ce, cw, cn, cs, ct, cb, cc, nx, ny, nz,   \
                 amb_temp, dt, Cap)
      for (uint64_t z = 1; z < nz - 1; z++) {
        for (uint64_t y = 1; y < ny - 1; y++) {
          /**
           * ! This will access one element out of the array,
           * ! but we leave it there to have perfect vectorization.
           */
          for (uint64_t x = 0; x < nx; x++) {
            uint64_t c = x + y * nx + z * nx * ny;
            float p = pIn[c];
            float tc = tIn_t[c];
            // float tw = (x == 0) ? tc : tIn_t[c - 1];
            // float te = (x == nx - 1) ? tc : tIn_t[c + 1];
            // float tn = (y == 0) ? tc : tIn_t[c - nx];
            // float ts = (y == ny - 1) ? tc : tIn_t[c + nx];
            // float tb = (z == 0) ? tc : tIn_t[c - nx * ny];
            // float tt = (z == nz - 1) ? tc : tIn_t[c + nx * ny];
            float tw = tIn_t[c - 1];
            float te = tIn_t[c + 1];
            float tn = tIn_t[c - nx];
            float ts = tIn_t[c + nx];
            float tb = tIn_t[c - nx * ny];
            float tt = tIn_t[c + nx * ny];
            tOut_t[c] = cc * tc + cw * tw + ce * te + cs * ts + cn * tn +
                        cb * tb + ct * tt + (dt / Cap) * p + ct * amb_temp;
          }
        }
      }
#ifdef GEM_FORGE
      m5_work_end(0, 0);
#endif
      float *t = tIn_t;
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

  char *pfile, *tfile, *ofile; // *testFile;

  pfile = argv[5];
  tfile = argv[6];
  ofile = argv[7];
  int numCols = atoi(argv[1]);
  int numRows = atoi(argv[1]);
  int layers = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  int numThreads = atoi(argv[4]);

  /* calculating parameters*/

  float dx = chip_height / numRows;
  float dy = chip_width / numCols;
  float dz = t_chip / layers;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx = dy / (2.0 * K_SI * t_chip * dx);
  float Ry = dx / (2.0 * K_SI * t_chip * dy);
  float Rz = dz / (K_SI * dx * dy);

  // cout << Rx << " " << Ry << " " << Rz << endl;
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt = PRECISION / max_slope;

  float *powerIn, *tempOut, *tempIn, *tempCopy; // *pCopy;
  //    float *d_powerIn, *d_tempIn, *d_tempOut;
  int size = numCols * numRows * layers;

  powerIn = (float *)calloc(size, sizeof(float));
  tempCopy = (float *)malloc(size * sizeof(float));
  tempIn = (float *)calloc(size, sizeof(float));
  tempOut = (float *)calloc(size, sizeof(float));
  // pCopy = (float*)calloc(size,sizeof(float));
  float *answer = (float *)calloc(size, sizeof(float));

  // outCopy = (float*)calloc(size, sizeof(float));
  readinput(powerIn, numRows, numCols, layers, pfile);
  readinput(tempIn, numRows, numCols, layers, tfile);

  memcpy(tempCopy, tempIn, size * sizeof(float));

  struct timeval start, stop;
  float time;
  gettimeofday(&start, NULL);

  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  printf("%d threads running\n", omp_get_num_threads());

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
#pragma omp parallel for schedule(static)                                      \
    firstprivate(powerIn, tempIn, tempOut, layers, numRows, numCols)
  for (uint64_t z = 0; z < layers; z++) {
    for (uint64_t y = 0; y < numRows; y++) {
      for (uint64_t x = 0; x < numCols; x += 64 / sizeof(float)) {
        uint64_t c = x + y * numCols + z * numCols * numRows;
        volatile float v1 = powerIn[c];
        volatile float v2 = tempIn[c];
        volatile float v3 = tempOut[c];
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
  gettimeofday(&stop, NULL);
  time = (stop.tv_usec - start.tv_usec) * 1.0e-6 + stop.tv_sec - start.tv_sec;
  computeTempCPU(powerIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx,
                 Ry, Rz, dt, iterations);

  float acc = accuracy(tempOut, answer, numRows * numCols * layers);
  printf("Time: %.3f (s)\n", time);
  printf("Accuracy: %e\n", acc);
  writeoutput(tempOut, numRows, numCols, layers, ofile);
  free(tempIn);
  free(tempOut);
  free(powerIn);
  return 0;
}
