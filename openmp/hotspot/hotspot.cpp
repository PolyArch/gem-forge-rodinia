#include <malloc.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#include <cassert>
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
// #define NUM_THREAD 4

#ifdef USE_FLOAT32
typedef float FLOAT;
#else
typedef double FLOAT;
#endif

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
void single_iteration(int threads, FLOAT *__restrict__ result,
                      FLOAT *__restrict__ temp, FLOAT *__restrict__ power,
                      int row, int col, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1,
                      FLOAT Rz_1, FLOAT step) {
#ifdef BLOCKED
  uint64_t num_chunk = row * col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
  uint64_t chunks_in_row = col / BLOCK_SIZE_C;
  uint64_t chunks_in_col = row / BLOCK_SIZE_R;
#endif

#ifdef GEM_FORGE
  m5_work_begin(0, 0);
#endif
#ifndef BLOCKED

#pragma omp parallel firstprivate(threads, result, temp, power, row, col,      \
                                      Cap_1, Rx_1, Ry_1, Rz_1, amb_temp)
  {

#ifdef SKEW_ROW

    // Rows are always interleaved by 1.
    int64_t rows_per_round = threads;
    int64_t row_rounds = (row + rows_per_round - 1) / rows_per_round;
    int64_t row_remainder = row % rows_per_round;
    if (row_remainder != 0) {
      exit(1);
    }

    int tid = omp_get_thread_num();

    // Skip the first row.
    int64_t rr_start = tid == 0 ? 1 : 0;
    // Skip the last row.
    int64_t rr_end = (tid == threads - 1) ? row_rounds - 1 : row_rounds;

#pragma clang loop unroll(disable) vectorize(disable)
    for (int64_t rr = rr_start; rr < rr_end; ++rr) {

      int64_t r = rr * rows_per_round + tid;

#else

#pragma omp for schedule(static)
    for (uint64_t r = 1; r < row - 1; ++r) {
#endif

      /**
       * ! This will access one element outside the array, but I keep it so
       * ! that the inner-most loop is perfectly vectorizable.
       */

#pragma omp simd
#ifdef GEM_FORGE_FIX_INPUT
      for (uint64_t c = 0; c < GEM_FORGE_FIX_INPUT_SIZE; ++c) {
        uint64_t idx = r * GEM_FORGE_FIX_INPUT_SIZE + c;
        uint64_t idxS = idx + GEM_FORGE_FIX_INPUT_SIZE;
        uint64_t idxN = idx - GEM_FORGE_FIX_INPUT_SIZE;
        // Just use some constant number to avoid too many inputs for stream
        // computation.
        FLOAT Cap = 0.5;
        FLOAT Rx = 1.5;
        FLOAT Ry = 1.2;
        FLOAT Rz = 0.7;
#else
      for (uint64_t c = 0; c < col; ++c) {
        uint64_t idx = r * col + c;
        uint64_t idxS = idx + col;
        uint64_t idxN = idx - col;
        FLOAT Cap = Cap_1;
        FLOAT Rx = Rx_1;
        FLOAT Ry = Ry_1;
        FLOAT Rz = Rz_1;
#endif
        uint64_t idxE = idx + 1;
        uint64_t idxW = idx - 1;
#pragma ss stream_name "rodinia.hotspot.power.ld"
        FLOAT powerC = power[idx];
#pragma ss stream_name "rodinia.hotspot.tempC.ld"
        FLOAT tempC = temp[idx];
#pragma ss stream_name "rodinia.hotspot.tempS.ld"
        FLOAT tempS = temp[idxS];
#pragma ss stream_name "rodinia.hotspot.tempN.ld"
        FLOAT tempN = temp[idxN];
#pragma ss stream_name "rodinia.hotspot.tempE.ld"
        FLOAT tempE = temp[idxE];
#pragma ss stream_name "rodinia.hotspot.tempW.ld"
        FLOAT tempW = temp[idxW];
        FLOAT delta = Cap * (powerC + (tempS + tempN - 2.f * tempC) * Ry +
                             (tempE + tempW - 2.f * tempC) * Rx +
                             (amb_temp - tempC) * Rz);
#pragma ss stream_name "rodinia.hotspot.result.st"
        result[idx] = tempC + delta;
      }
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

#ifdef GEM_FORGE
void gf_warm_array(const char *name, void *buffer, uint64_t totalBytes) {
  uint64_t cachedBytes = m5_stream_nuca_get_cached_bytes(buffer);
  printf("[GF_WARM] Region %s TotalBytes %lu CachedBytes %lu Cached %.2f%%.\n",
         name, totalBytes, cachedBytes,
         static_cast<float>(cachedBytes) / static_cast<float>(totalBytes) *
             100.f);
  assert(cachedBytes <= totalBytes);
  for (uint64_t i = 0; i < cachedBytes; i += 64) {
    __attribute__((unused)) volatile uint8_t data =
        reinterpret_cast<uint8_t *>(buffer)[i];
  }
  printf("[GF_WARM] Region %s Warmed %.2f%%.\n", name,
         static_cast<float>(cachedBytes) / static_cast<float>(totalBytes) *
             100.f);
}
#endif

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp,
                       FLOAT *power, int row, int col, int warm) {
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
#endif
  if (warm) {
#ifdef GEM_FORGE
    gf_warm_array("temp", temp, sizeof(temp[0]) * row * col);
    gf_warm_array("result", result, sizeof(result[0]) * row * col);
    gf_warm_array("power", power, sizeof(power[0]) * row * col);
#else
    for (uint64_t i = 0; i < row * col; i += 64 / sizeof(FLOAT)) {
      volatile FLOAT vr = result[i];
    }
    for (uint64_t i = 0; i < row * col; i += 64 / sizeof(FLOAT)) {
      volatile FLOAT vr = temp[i];
    }
    for (uint64_t i = 0; i < row * col; i += 64 / sizeof(FLOAT)) {
      volatile FLOAT vr = power[i];
    }
#endif
  }
  // Start the threads.
#pragma omp parallel for schedule(static)
  for (uint64_t r = 0; r < num_omp_threads; ++r) {
    volatile FLOAT vr = result[r];
  }
#ifdef GEM_FORGE
  m5_reset_stats(0, 0);
#endif

  FLOAT *r = result;
  FLOAT *t = temp;
  for (int i = 0; i < num_iterations; i++) {
    single_iteration(num_omp_threads, r, t, power, row, col, Cap_1, Rx_1, Ry_1,
                     Rz_1, step);
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
          "threads> <temp_file> <power_file> <output_file> <warm>\n",
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
  fprintf(stderr, "\t<warm> - Whether to warm up the cache\n");
  exit(1);
}

int main(int argc, char **argv) {
  int grid_rows, grid_cols, sim_time;
  char *tfile, *pfile, *ofile;

  /* check validity of inputs	*/
  if (argc != 9)
    usage(argc, argv);
  if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
      (sim_time = atoi(argv[3])) <= 0 || (num_omp_threads = atoi(argv[4])) <= 0)
    usage(argc, argv);

  int warm = atoi(argv[8]);

#ifdef GEM_FORGE_FIX_INPUT
  if (grid_cols != GEM_FORGE_FIX_INPUT_SIZE) {
    printf("Input size mismatch, fixed %d.\n", GEM_FORGE_FIX_INPUT_SIZE);
    exit(1);
  }
#endif

  // Make result offset by some pages.
#ifndef OFFSET_BYTES
#define OFFSET_BYTES 0
#endif
  const int OFFSET_ELEMENTS = OFFSET_BYTES / sizeof(FLOAT);
  const int PAGE_SIZE = 4096;
  const int CACHE_BLOCK_SIZE = 64;
  const int size = grid_rows * grid_cols;
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
  FLOAT *temp = Buffer + 0;
  FLOAT *power = Buffer + size;
  FLOAT *result = Buffer + size + size + OFFSET_ELEMENTS;

  // Now we touch all the pages according to the index.
  int elementsPerPage = PAGE_SIZE / sizeof(FLOAT);
#pragma clang loop vectorize(disable) unroll(disable) interleave(disable)
  for (int i = 0; i < numPages; i++) {
    int pageIdx = idx[i];
    int elementIdx = pageIdx * elementsPerPage;
    volatile FLOAT v = Buffer[elementIdx];
  }

#ifdef GEM_FORGE
  // Stream SNUCA.
  m5_stream_nuca_region("rodinia.hotspot.temp", temp, sizeof(temp[0]),
                        grid_cols, grid_rows, 0);
  m5_stream_nuca_region("rodinia.hotspot.power", power, sizeof(power[0]),
                        grid_cols, grid_rows, 0);
  m5_stream_nuca_region("rodinia.hotspot.result", result, sizeof(result[0]),
                        grid_cols, grid_rows, 0);
  m5_stream_nuca_align(power, power, grid_cols);
  m5_stream_nuca_align(temp, power, 0);
  m5_stream_nuca_align(result, power, 0);
  m5_stream_nuca_remap();
#endif

  if (!temp || !power) {
    printf("unable to allocate memory");
    exit(1);
  }
  printf("Size of array %luMB.\n",
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

  compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols, warm);
  /* cleanup	*/
  // free(temp);
  // free(power);
  free(Buffer);

  return 0;
}
