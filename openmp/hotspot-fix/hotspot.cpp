#include <malloc.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

// Returns the current system time in microseconds
#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5
#define OPEN
//#define NUM_THREAD 4

typedef double FLOAT;

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */
void single_iteration(FLOAT *__restrict__ result, FLOAT *__restrict__ temp,
                      FLOAT *__restrict__ power, int row, int col, FLOAT Cap_1,
                      FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1, FLOAT step) {
#ifdef BLOCKED
  uint64_t num_chunk = row * col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
  uint64_t chunks_in_row = col / BLOCK_SIZE_C;
  uint64_t chunks_in_col = row / BLOCK_SIZE_R;
#endif

#ifdef GEM_FORGE
  m5_work_begin(0, 0);
#endif
#ifndef BLOCKED
#pragma omp parallel for shared(power, temp, result)                           \
    firstprivate(row, col, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp) schedule(static)
  for (uint64_t r = 1; r < row - 1; ++r) {
    /**
     * ! This will access one element outside the array, but I keep it so that
     * ! the inner-most loop is perfectly vectorizable.
     */
#ifdef GEM_FORGE_FIX_INPUT
#pragma omp simd
    for (uint64_t c = 0; c < GEM_FORGE_FIX_INPUT_SIZE; ++c) {
      uint64_t idx = r * GEM_FORGE_FIX_INPUT_SIZE + c;
      uint64_t idxS = idx + GEM_FORGE_FIX_INPUT_SIZE;
      uint64_t idxN = idx - GEM_FORGE_FIX_INPUT_SIZE;
#else
#pragma omp simd
    for (uint64_t c = 0; c < col; ++c) {
      uint64_t idx = r * col + c;
      uint64_t idxS = idx + col;
      uint64_t idxN = idx - col;
#endif
      uint64_t idxE = idx + 1;
      uint64_t idxW = idx - 1;
      FLOAT powerC = power[idx];
      FLOAT tempC = temp[idx];
      FLOAT tempS = temp[idxS];
      FLOAT tempN = temp[idxN];
      FLOAT tempE = temp[idxE];
      FLOAT tempW = temp[idxW];
      FLOAT delta = Cap_1 * (powerC + (tempS + tempN - 2.f * tempC) * Ry_1 +
                             (tempE + tempW - 2.f * tempC) * Rx_1 +
                             (amb_temp - tempC) * Rz_1);
      result[idx] = tempC + delta;
    }
  }
#else
#pragma omp parallel for shared(power, temp, result)                           \
    firstprivate(row, col, num_chunk, chunks_in_row, chunks_in_col, Cap_1,     \
                 Rx_1, Ry_1, Rz_1, amb_temp) schedule(static)
  for (uint64_t chunk = 0; chunk < num_chunk; ++chunk) {
    uint64_t r_start = BLOCK_SIZE_R * (chunk / chunks_in_col);
    uint64_t c_start = BLOCK_SIZE_C * (chunk % chunks_in_row);
    uint64_t r_end = r_start + BLOCK_SIZE_R;
    uint64_t c_end = c_start + BLOCK_SIZE_C;

    for (uint64_t r = r_start; r < r_end; ++r) {
      for (uint64_t c = c_start; c < c_end; ++c) {
        uint64_t idx = r * col + c;
        uint64_t idxE = idx + 1;
        uint64_t idxW = idx - 1;
        uint64_t idxS = idx + col;
        uint64_t idxN = idx - col;
        FLOAT powerC = power[idx];
        FLOAT tempC = temp[idx];
        FLOAT tempN = tempC;
        FLOAT tempS = tempC;
        FLOAT tempE = tempC;
        FLOAT tempW = tempC;
        if (r < row - 1) {
          tempS = temp[idx + col];
        }
        if (r > 0) {
          tempN = temp[idx - col];
        }
        if (c < col - 1) {
          tempE = temp[idx + 1];
        }
        if (c > 0) {
          tempW = temp[idx - 1];
        }
        FLOAT delta = Cap_1 * (powerC + (tempS + tempN - 2.f * tempC) * Ry_1 +
                               (tempE + tempW - 2.f * tempC) * Rx_1 +
                               (amb_temp - tempC) * Rz_1);
        result[idx] = tempC + delta;
      }
    }
  }
#endif

#ifdef GEM_FORGE
  m5_work_end(0, 0);
#endif
}

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp,
                       FLOAT *power, int row, int col) {
  FLOAT grid_height = chip_height / row;
  FLOAT grid_width = chip_width / col;

  FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

  FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  FLOAT step = PRECISION / max_slope / 1000.0;

  FLOAT Rx_1 = 1.f / Rx;
  FLOAT Ry_1 = 1.f / Ry;
  FLOAT Rz_1 = 1.f / Rz;
  FLOAT Cap_1 = step / Cap;

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
#pragma omp parallel for shared(power, temp, result) firstprivate(row, col)    \
    schedule(static)
  for (uint64_t r = 0; r < row; ++r) {
    for (uint64_t c = 0; c < col; c += 64 / sizeof(FLOAT)) {
      int idx = r * col + c;
      volatile FLOAT vr = result[idx];
      volatile FLOAT vt = temp[idx];
      volatile FLOAT vp = power[idx];
    }
  }
  m5_reset_stats(0, 0);
#endif
#endif

  FLOAT *r = result;
  FLOAT *t = temp;
  for (int i = 0; i < num_iterations; i++) {
    single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
    FLOAT *tmp = t;
    t = r;
    r = tmp;
  }

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
}

void fatal(char *s) {
  fprintf(stderr, "error: %s\n", s);
  exit(1);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
  int i, index;
  FILE *fp;
  char str[STR_SIZE];
  FLOAT val;

  fp = fopen(file, "r");
  if (!fp) {
    printf("file could not be opened for reading");
    exit(1);
  }

  fread(vect, sizeof(float), grid_rows * grid_cols, fp);

  fclose(fp);
}

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of "
          "threads><temp_file> <power_file>\n",
          argv[0]);
  fprintf(stderr,
          "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
  fprintf(
      stderr,
      "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<no. of threads>   - number of threads\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial "
                  "temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated "
                  "power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char **argv) {
  int grid_rows, grid_cols, sim_time, i;
  FLOAT *temp, *power, *result;
  char *tfile, *pfile, *ofile;

  /* check validity of inputs	*/
  if (argc != 8)
    usage(argc, argv);
  if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
      (sim_time = atoi(argv[3])) <= 0 || (num_omp_threads = atoi(argv[4])) <= 0)
    usage(argc, argv);

#ifdef GEM_FORGE_FIX_INPUT
  if (grid_cols != GEM_FORGE_FIX_INPUT_SIZE) {
    printf("Input size mismatch, fixed %d.\n", GEM_FORGE_FIX_INPUT_SIZE);
    exit(1);
  }
#endif

  /* allocate memory for the temperature and power arrays	*/
  temp = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
  power = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
  result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
  if (!temp || !power) {
    printf("unable to allocate memory");
    exit(1);
  }
  printf("Size of array %dMB.\n",
         3 * grid_rows * grid_cols * sizeof(FLOAT) / 1024 / 1024);

  /* read initial temperatures and input power	*/
  tfile = argv[5];
  pfile = argv[6];
  ofile = argv[7];

  // read_input(temp, grid_rows, grid_cols, tfile);
  // read_input(power, grid_rows, grid_cols, pfile);

  printf("Start computing the transient temperature\n");

  omp_set_num_threads(num_omp_threads);
#ifdef GEM_FORGE
  // mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

  compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);
  /* cleanup	*/
  free(temp);
  free(power);

  return 0;
}
