#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

#include <omp.h>

#include "timer.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

void run(int argc, char **argv);

/* define timer macros */
#define pin_stats_reset() startCycle()
#define pin_stats_pause(cycles) stopCycle(cycles)
#define pin_stats_dump(cycles) printf("timer: %Lu\n", cycles)

// #define BENCH_PRINT

int64_t rows, cols;
int num_threads;
int *data;
int *wall;
int *result;
#define M_SEED 9

void init(int argc, char **argv) {
  if (argc == 4) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    num_threads = atoi(argv[3]);
  } else {
    printf("Usage: pathfiner width num_of_steps num_of_threads\n");
    exit(0);
  }
  wall = new int[rows * cols];
  result = new int[cols + 2];

  int seed = M_SEED;
  srand(seed);

#ifndef GEM_FORGE
  // No need to initialize as it's data independent.
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i * cols + j] = rand() % 10;
    }
  }
  for (int j = 0; j < cols; j++)
    result[j + 1] = wall[0 * cols + j];
#endif
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", wall[i * cols + j]);
    }
    printf("\n");
  }
#endif
}

void fatal(char *s) { fprintf(stderr, "error: %s\n", s); }

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

int main(int argc, char **argv) {
  run(argc, argv);

  return EXIT_SUCCESS;
}

__attribute__((noinline)) void pathfinder(int *src, int *dst) {
  for (int64_t t = 0; t < rows - 1; t++) {
    int *temp = src;
    src = dst;
    dst = temp;

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#pragma omp parallel for firstprivate(cols, t) schedule(static)
    for (int64_t n = 0; n < cols; n++) {
      int min = MIN(src[n], MIN(src[n + 1], src[n + 2]));
      int w = wall[t * cols + n];
      dst[n + 1] = w + min;
    }

    // Expand the boundary values.
    dst[0] = dst[1];
    dst[cols + 1] = dst[cols];

#ifdef GEM_FORGE
    m5_work_end(0, 0);
#endif
  }
}

void run(int argc, char **argv) {
  init(argc, argv);

  unsigned long long cycles;

  int *src, *dst, *temp;
  int min;

  dst = result;
  src = new int[cols + 2];

  pin_stats_reset();

  printf("Running with threads %d.\n", num_threads);
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  omp_set_schedule(omp_sched_static, 0);
#ifdef GEM_FORGE
  mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();

/**
 * In Ruby, cache is disabled in fast forward mode.
 * However, this program only run once, so we manually
 * warm up the cache by iterating through the wall array
 * and reset stats after that.
 * Of course this will make the simulation time longer.
 */
#ifdef GEM_FORGE_WARM_CACHE
  for (int t = 0; t < rows; ++t) {
#pragma omp parallel for firstprivate(t, cols, wall, src, dst) schedule(static)
    for (int n = 0; n < cols; n += (64 / sizeof(int))) {
      volatile int v = wall[t * cols + n];
      volatile int vs = src[n];
      volatile int vd = dst[n];
    }
  }
#endif
  m5_reset_stats(0, 0);

#endif

  pathfinder(src, dst);
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

  pin_stats_pause(cycles);
  pin_stats_dump(cycles);

#ifdef BENCH_PRINT

  for (int i = 0; i < cols; i++)

    printf("%d ", data[i]);

  printf("\n");

  for (int i = 0; i < cols; i++)

    printf("%d ", dst[i]);

  printf("\n");

#endif

  delete[] data;
  delete[] wall;
  delete[] dst;
  delete[] src;
}
