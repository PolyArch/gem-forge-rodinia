#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
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

  int count = 0;
  FLOAT *tIn_t = tIn;
  FLOAT *tOut_t = tOut;

  do {
#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif
    /**
     * ! This will access one element out of the array,
     * ! but we leave it there to have perfect vectorization.
     */

#if defined(FIX_ROW) && defined(FIX_COL)
#define ROW_SIZE FIX_ROW
#define COL_SIZE FIX_COL
#define MAT_SIZE (ROW_SIZE * COL_SIZE)

#ifdef FUSE_OUTER_LOOPS

#define START_ROW ROW_SIZE
    uint64_t END_ROW = (nz - 1) * ROW_SIZE;

#ifdef GEM_FORGE_DYN_SCHEDULE
#pragma omp parallel for schedule(dynamic, GEM_FORGE_DYN_SCHEDULE)             \
    firstprivate(tIn_t, pIn, tOut_t)
#else
#pragma omp parallel for schedule(static) firstprivate(tIn_t, pIn, tOut_t)
#endif // GEM_FORGE_DYN_SCHEDULE
    for (uint64_t row = START_ROW; row < END_ROW; ++row) {
#pragma omp simd
      for (uint64_t x = 0; x < COL_SIZE; x++) {
        uint64_t c = x + row * COL_SIZE;
#else
#pragma omp parallel for schedule(static) firstprivate(nz, tIn_t, pIn, tOut_t)
    for (uint64_t z = 1; z < nz - 1; z++) {
      for (uint64_t y = 0; y < ROW_SIZE; y++) {
#pragma omp simd
        for (uint64_t x = 0; x < COL_SIZE; x++) {
          uint64_t c = x + y * COL_SIZE + z * MAT_SIZE;
#endif // FUSE_OUTER_LOOPS

        uint64_t idxT = c - MAT_SIZE;
        uint64_t idxB = c + MAT_SIZE;
        uint64_t idxN = c - COL_SIZE;
        uint64_t idxS = c + COL_SIZE;
        // Use fixed immediate.
        FLOAT localCC = 0.5;
        FLOAT localCE = 0.6;
        FLOAT localCW = 0.2;
        FLOAT localCN = 0.3;
        FLOAT localCS = 0.4;
        FLOAT localCT = 0.8;
        FLOAT localCB = 0.7;
        FLOAT localdt = 0.1;
        FLOAT localCap = 0.8;
        FLOAT localAmb = 0.7;
        uint64_t idxW = c - 1;
        uint64_t idxE = c + 1;
#pragma ss stream_name "rodinia.hotspot3D.power.ld"
        FLOAT p = pIn[c];
#pragma ss stream_name "rodinia.hotspot3D.tc.ld"
        FLOAT tc = tIn_t[c];
#pragma ss stream_name "rodinia.hotspot3D.tw.ld"
        FLOAT tw = tIn_t[idxW];
#pragma ss stream_name "rodinia.hotspot3D.te.ld"
        FLOAT te = tIn_t[idxE];
#pragma ss stream_name "rodinia.hotspot3D.tn.ld"
        FLOAT tn = tIn_t[idxN];
#pragma ss stream_name "rodinia.hotspot3D.ts.ld"
        FLOAT ts = tIn_t[idxS];
#pragma ss stream_name "rodinia.hotspot3D.tb.ld"
        FLOAT tb = tIn_t[idxB];
#pragma ss stream_name "rodinia.hotspot3D.tt.ld"
        FLOAT tt = tIn_t[idxT];
#pragma ss stream_name "rodinia.hotspot3D.out.st"
        tOut_t[c] = localCC * tc + localCW * tw + localCE * te + localCS * ts +
                    localCN * tn + localCB * tb + localCT * tt +
                    (localdt / localCap) * p + localCT * localAmb;
      }
    }
#ifndef FUSE_OUTER_LOOPS
  }
#endif

#else
#pragma omp parallel for schedule(static)                                      \
    firstprivate(tIn_t, pIn, tOut_t, ce, cw, cn, cs, ct, cb, cc, nx, ny, nz,   \
                 amb_temp, dt, Cap)
    for (uint64_t z = 1; z < nz - 1; z++) {
      for (uint64_t y = 0; y < ny; y++) {
#pragma omp simd
        for (uint64_t x = 0; x < nx; x++) {
          uint64_t c = x + y * nx + z * nx * ny;
          uint64_t idxT = c - nx * ny;
          uint64_t idxB = c + nx * ny;
          uint64_t idxN = c - nx;
          uint64_t idxS = c + nx;
          FLOAT localCC = cc;
          FLOAT localCE = ce;
          FLOAT localCW = cw;
          FLOAT localCN = cn;
          FLOAT localCS = cs;
          FLOAT localCT = ct;
          FLOAT localCB = cb;
          FLOAT localdt = dt;
          FLOAT localCap = Cap;
          FLOAT localAmb = amb_temp;
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
          tOut_t[c] = localCC * tc + localCW * tw + localCE * te +
                      localCS * ts + localCN * tn + localCB * tb +
                      localCT * tt + (localdt / localCap) * p +
                      localCT * localAmb;
        }
      }
    }
#endif

#ifdef GEM_FORGE
  m5_work_end(0, 0);
#endif
  FLOAT *t = tIn_t;
  tIn_t = tOut_t;
  tOut_t = t;
  count++;
}
while (count < numiter)
  ;
return;
}

void usage(int argc, char **argv) {
  fprintf(
      stderr,
      "Usage: %s <rows> <cols> <layers> <iterations> <nthreads> <powerFile> "
      "<tempFile> <outputFile> <warm>\n",
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
  fprintf(stderr, "\t<outputFile> - output file\n");
  fprintf(stderr, "\t<warm> - whether to warm up the cache\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 10) {
    usage(argc, argv);
  }

  int numRows = atoi(argv[1]);
  int numCols = atoi(argv[2]);
  int layers = atoi(argv[3]);
  int iterations = atoi(argv[4]);
  int numThreads = atoi(argv[5]);
  char *pfile = argv[6];
  char *tfile = argv[7];
  char *ofile = argv[8];
  int warm = atoi(argv[9]);

#ifndef FUSE_OUTER_LOOPS
  if (numThreads + 2 > layers) {
    numThreads = (layers > 2) ? (layers - 2) : 1;
  }
#endif

#if defined(FIX_ROW) && defined(FIX_COL)
  if (numCols != FIX_COL || numRows != FIX_ROW) {
    printf("Mismatch Fixed Dimension %dx%d != Input %dx%d.\n", FIX_ROW, FIX_COL,
           numRows, numCols);
    assert(0);
  }
#endif

  /* calculating parameters*/

  FLOAT dx = chip_height / numRows;
  FLOAT dy = chip_width / numCols;
  FLOAT dz = t_chip / layers;

  FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  FLOAT Rx = dy / (2.0 * K_SI * t_chip * dx);
  FLOAT Ry = dx / (2.0 * K_SI * t_chip * dy);
  FLOAT Rz = dz / (K_SI * dx * dy);

  FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  FLOAT dt = PRECISION / max_slope;

  int size = numCols * numRows * layers;

  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
  printf("Size %dx%dx%d. Threads %d\n", numRows, numCols, layers, numThreads);

  // Make tempOut offset by some pages.
#ifndef OFFSET_BYTES
#define OFFSET_BYTES 0
#endif
  const int OFFSET_ELEMENTS = OFFSET_BYTES / sizeof(FLOAT);
  const int PAGE_SIZE = 4096;
  const int CACHE_BLOCK_SIZE = 64;
  int totalBytes = 3 * size * sizeof(FLOAT) + OFFSET_BYTES;
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
  FLOAT *Buffer = (FLOAT *)aligned_alloc(PAGE_SIZE, numPages * PAGE_SIZE);
  FLOAT *powerIn = Buffer + 0;
  FLOAT *tempIn = Buffer + size;
  FLOAT *tempOut = Buffer + size + size + OFFSET_ELEMENTS;

  // Now we touch all the pages according to the index.
  int elementsPerPage = PAGE_SIZE / sizeof(FLOAT);
#pragma clang loop vectorize(disable) unroll(disable) interleave(disable)
  for (int i = 0; i < numPages; i++) {
    int pageIdx = idx[i];
    int elementIdx = pageIdx * elementsPerPage;
    volatile FLOAT v = Buffer[elementIdx];
  }

#ifdef GEM_FORGE
  m5_stream_nuca_region(powerIn, sizeof(powerIn[0]), size);
  m5_stream_nuca_region(tempIn, sizeof(tempIn[0]), size);
  m5_stream_nuca_region(tempOut, sizeof(tempOut[0]), size);
  m5_stream_nuca_align(powerIn, powerIn, numCols);
  m5_stream_nuca_align(powerIn, powerIn, numCols * numRows);
  m5_stream_nuca_align(tempIn, powerIn, 0);
  m5_stream_nuca_align(tempOut, powerIn, 0);
  m5_stream_nuca_remap();
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

  if (warm) {
    /**
     * Warm them up separately.
     */
#define WARM_ARRAY(A)                                                          \
  for (int64_t i = 0; i < size; i += 64 / sizeof(FLOAT)) {                     \
    volatile FLOAT v = A[i];                                                   \
  }
    WARM_ARRAY(powerIn);
    WARM_ARRAY(tempIn);
    WARM_ARRAY(tempOut);
#undef WARM_ARRAY
  }

  // Start the threads.
#pragma omp parallel for schedule(static)
  for (uint64_t z = 0; z < numThreads; z++) {
    volatile FLOAT v1 = powerIn[z];
  }

#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  computeTempOMP(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx,
                 Ry, Rz, dt, iterations);
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
  free(Buffer);
  // free(tempIn);
  // free(tempOut);
}
