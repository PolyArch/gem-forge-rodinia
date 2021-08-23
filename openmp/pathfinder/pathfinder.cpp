#include <assert.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
int *wall;
int *result;
int *temp;
int *Buffer;
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
  // wall = reinterpret_cast<int *>(aligned_alloc(64, rows * cols *
  // sizeof(int))); temp = reinterpret_cast<int *>(aligned_alloc(64, (cols + 2)
  // * sizeof(int))); result = reinterpret_cast<int *>(aligned_alloc(64, (cols +
  // 2) * sizeof(int)));

  // Make result offset by some pages.
#ifndef OFFSET_BYTES
#define OFFSET_BYTES 0
#endif
  const int OFFSET_ELEMENTS = OFFSET_BYTES / sizeof(int);
  const int PAGE_SIZE = 4096;
  const int CACHE_BLOCK_SIZE = 64;
  const int64_t size = rows * cols;
  int64_t totalBytes = (size + 2 * cols + 4) * sizeof(int) + OFFSET_BYTES;
  int64_t numPages = (totalBytes + PAGE_SIZE - 1) / PAGE_SIZE;
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
  Buffer = (int *)aligned_alloc(PAGE_SIZE, numPages * PAGE_SIZE);
  wall = Buffer + 0;
  temp = Buffer + size;
  result = Buffer + size + cols + OFFSET_ELEMENTS;

  // Now we touch all the pages according to the index.
  int elementsPerPage = PAGE_SIZE / sizeof(int);
#pragma clang loop vectorize(disable) unroll(disable) interleave(disable)
  for (int i = 0; i < numPages; i++) {
    int pageIdx = idx[i];
    int elementIdx = pageIdx * elementsPerPage;
    volatile int v = Buffer[elementIdx];
  }

#ifdef GEM_FORGE
  // Stream SNUCA.
  m5_stream_nuca_region(wall, sizeof(wall[0]), size);
  m5_stream_nuca_region(temp, sizeof(temp[0]), cols);
  m5_stream_nuca_region(result, sizeof(result[0]), cols);
  m5_stream_nuca_align(wall, wall, cols);
  m5_stream_nuca_align(temp, wall, 0);
  m5_stream_nuca_align(result, wall, 0);
  m5_stream_nuca_remap();
#endif

#ifndef GEM_FORGE
  // No need to initialize as it's data independent.
  int seed = M_SEED;
  srand(seed);
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

  int min;

  int *dst = result;
  int *src = temp;

  pin_stats_reset();

  printf("Running with threads %d.\n", num_threads);
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  omp_set_schedule(omp_sched_static, 0);
#ifdef GEM_FORGE
  // mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();

#ifdef GEM_FORGE_WARM_CACHE
  // Touch them to warm up.
  for (int64_t n = 0; n < rows * cols; n += (64 / sizeof(int))) {
    volatile int v = wall[n];
  }
  for (int64_t n = 0; n < cols; n += (64 / sizeof(int))) {
    volatile int vs = src[n];
  }
  for (int64_t n = 0; n < cols; n += (64 / sizeof(int))) {
    volatile int vd = dst[n];
  }
#pragma omp parallel for firstprivate(wall) schedule(static)
  for (int n = 0; n < num_threads; n++) {
    volatile int v = wall[n];
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

  free(wall);
  free(dst);
  free(src);
}
