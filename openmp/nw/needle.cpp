#define LIMIT -999
//#define TRACE
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define OPENMP

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// Returns the current system time in microseconds
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
int maximum(int a, int b, int c) {

  int k;
  if (a <= b)
    k = b;
  else
    k = a;

  if (k <= c)
    return (c);
  else
    return (k);
}
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif

#include "blosum64.h"

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  runTest(argc, argv);

  return EXIT_SUCCESS;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n",
          argv[0]);
  fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
  fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
  fprintf(stderr, "\t<num_threads>    - no. of threads\n");
  exit(1);
}

void nw_optimized(int *input_itemsets, int *output_itemsets, int *referrence,
                  uint64_t max_rows, uint64_t max_cols, int penalty) {
#ifdef OMP_OFFLOAD
  int transfer_size = max_rows * max_cols;
#pragma omp target data map(to                                                 \
                            : max_cols, penalty, referrence [0:transfer_size]) \
    map(input_itemsets [0:transfer_size])
  {

#pragma omp target
#endif

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

    for (uint64_t blk = 1; blk <= (max_cols - 1) / BLOCK_SIZE; blk++) {
#ifdef OPENMP
#pragma omp parallel for schedule(static) shared(input_itemsets, referrence)   \
    firstprivate(blk, max_rows, max_cols, penalty)
#endif
      for (uint64_t b_index_x = 0; b_index_x < blk; ++b_index_x) {
        uint64_t b_index_y = blk - 1 - b_index_x;
        // Compute
        for (uint64_t i = 1; i < BLOCK_SIZE + 1; ++i) {
          for (uint64_t j = 1; j < BLOCK_SIZE + 1; ++j) {
            uint64_t idx = (i + b_index_y * BLOCK_SIZE) * max_cols +
                           b_index_x * BLOCK_SIZE + j;
            uint64_t idxNW = idx - max_cols - 1;
            uint64_t idxN = idx - max_cols;
            uint64_t idxW = idx - 1;

            input_itemsets[idx] = maximum(
                input_itemsets[idxNW] + referrence[idx],
                input_itemsets[idxW] - penalty, input_itemsets[idxN] - penalty);
          }
        }
      }
    }

#ifdef GEM_FORGE
    m5_work_end(0, 0);
#endif

#ifdef GEM_FORGE
    m5_work_begin(1, 0);
#endif

#ifdef OMP_OFFLOAD
#pragma omp target
#endif
    for (uint64_t blk = 2; blk <= (max_cols - 1) / BLOCK_SIZE; blk++) {
#ifdef OPENMP
#pragma omp parallel for schedule(static) shared(input_itemsets, referrence)   \
    firstprivate(blk, max_rows, max_cols, penalty)
#endif
      for (uint64_t b_index_x = blk - 1;
           b_index_x < (max_cols - 1) / BLOCK_SIZE; ++b_index_x) {
        uint64_t b_index_y = (max_cols - 1) / BLOCK_SIZE + blk - 2 - b_index_x;
        /**
         * ! I avoid using local array.
         */
        for (uint64_t i = 1; i < BLOCK_SIZE + 1; ++i) {
          for (uint64_t j = 1; j < BLOCK_SIZE + 1; ++j) {
            uint64_t idx = (i + b_index_y * BLOCK_SIZE) * max_cols +
                           b_index_x * BLOCK_SIZE + j;
            uint64_t idxNW = idx - max_cols - 1;
            uint64_t idxN = idx - max_cols;
            uint64_t idxW = idx - 1;

            input_itemsets[idx] = maximum(
                input_itemsets[idxNW] + referrence[idx],
                input_itemsets[idxW] - penalty, input_itemsets[idxN] - penalty);
          }
        }

        //         int input_itemsets_l[(BLOCK_SIZE + 1) * (BLOCK_SIZE + 1)]
        //             __attribute__((aligned(64)));
        //         int reference_l[BLOCK_SIZE * BLOCK_SIZE]
        //         __attribute__((aligned(64)));

        //         // Copy referrence to local memory
        //         for (uint64_t i = 0; i < BLOCK_SIZE; ++i) {
        // #pragma omp simd
        //           for (uint64_t j = 0; j < BLOCK_SIZE; ++j) {
        //             reference_l[i * BLOCK_SIZE + j] =
        //                 referrence[max_cols * (b_index_y * BLOCK_SIZE + i +
        //                 1) +
        //                            b_index_x * BLOCK_SIZE + j + 1];
        //           }
        //         }

        //         // Copy input_itemsets to local memory
        //         for (uint64_t i = 0; i < BLOCK_SIZE + 1; ++i) {
        // #pragma omp simd
        //           for (uint64_t j = 0; j < BLOCK_SIZE + 1; ++j) {
        //             input_itemsets_l[i * (BLOCK_SIZE + 1) + j] =
        //                 input_itemsets[max_cols * (b_index_y * BLOCK_SIZE +
        //                 i) +
        //                                b_index_x * BLOCK_SIZE + j];
        //           }
        //         }

        //         // Compute
        //         for (uint64_t i = 1; i < BLOCK_SIZE + 1; ++i) {
        //           for (uint64_t j = 1; j < BLOCK_SIZE + 1; ++j) {
        //             input_itemsets_l[i * (BLOCK_SIZE + 1) + j] = maximum(
        //                 input_itemsets_l[(i - 1) * (BLOCK_SIZE + 1) + j - 1]
        //                 +
        //                     reference_l[(i - 1) * BLOCK_SIZE + j - 1],
        //                 input_itemsets_l[i * (BLOCK_SIZE + 1) + j - 1] -
        //                 penalty, input_itemsets_l[(i - 1) * (BLOCK_SIZE + 1)
        //                 + j] - penalty);
        //           }
        //         }

        //         // Copy results to global memory
        //         for (uint64_t i = 0; i < BLOCK_SIZE; ++i) {
        // #pragma omp simd
        //           for (uint64_t j = 0; j < BLOCK_SIZE; ++j) {
        //             input_itemsets[max_cols * (b_index_y * BLOCK_SIZE + i +
        //             1) +
        //                            b_index_x * BLOCK_SIZE + j + 1] =
        //                 input_itemsets_l[(i + 1) * (BLOCK_SIZE + 1) + j + 1];
        //           }
        //         }
      }
    }

#ifdef GEM_FORGE
    m5_work_end(1, 0);
#endif

#ifdef OMP_OFFLOAD
  }
#endif
}

void runTest(int argc, char **argv) {
  int max_rows, max_cols, penalty;
  int *input_itemsets, *output_itemsets, *referrence;
  // int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
  // int size;
  int omp_num_threads;

  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  if (argc == 4) {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
    omp_num_threads = atoi(argv[3]);
  } else {
    usage(argc, argv);
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
  input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
  output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

  if (!input_itemsets)
    fprintf(stderr, "error: can not allocate memory");

  srand(7);

  for (uint64_t i = 0; i < max_cols; i++) {
    for (uint64_t j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  printf("Start Needleman-Wunsch\n");

  for (uint64_t i = 1; i < max_rows; i++) { // please define your own sequence.
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  }
  for (uint64_t j = 1; j < max_cols; j++) { // please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (uint64_t i = 1; i < max_cols; i++) {
    for (uint64_t j = 1; j < max_rows; j++) {
      referrence[i * max_cols + j] =
          blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (uint64_t i = 1; i < max_rows; i++)
    input_itemsets[i * max_cols] = -i * penalty;
  for (uint64_t j = 1; j < max_cols; j++)
    input_itemsets[j] = -j * penalty;

  // Compute top-left matrix
  omp_set_dynamic(0);
  omp_set_num_threads(omp_num_threads);
  printf("Num of threads: %d\n", omp_get_num_threads());
  printf("Processing top-left matrix\n");

  long long start_time = get_time();

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  for (uint64_t i = 0; i < max_rows; ++i) {
    for (uint64_t j = 0; j < max_cols; j += 64 / sizeof(int)) {
      uint64_t idx = i * max_cols + j;
      volatile int vi = input_itemsets[idx];
      volatile int vo = output_itemsets[idx];
      volatile int vr = referrence[idx];
    }
  }
  m5_reset_stats(0, 0);
#endif
#endif
  nw_optimized(input_itemsets, output_itemsets, referrence, max_rows, max_cols,
               penalty);
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

  long long end_time = get_time();

  printf("Total time: %.3f seconds\n",
         ((float)(end_time - start_time)) / (1000 * 1000));

// #define TRACEBACK
#ifdef TRACEBACK

  FILE *fpo = fopen("result.txt", "w");
  fprintf(fpo, "print traceback value GPU:\n");

  for (uint64_t i = max_rows - 2, j = max_rows - 2; i >= 0 && j >= 0;) {
    int nw, n, w, traceback;
    if (i == max_rows - 2 && j == max_rows - 2)
      fprintf(fpo, "%d ",
              input_itemsets[i * max_cols + j]); // print the first element
    if (i == 0 && j == 0)
      break;
    if (i > 0 && j > 0) {
      nw = input_itemsets[(i - 1) * max_cols + j - 1];
      w = input_itemsets[i * max_cols + j - 1];
      n = input_itemsets[(i - 1) * max_cols + j];
    } else if (i == 0) {
      nw = n = LIMIT;
      w = input_itemsets[i * max_cols + j - 1];
    } else if (j == 0) {
      nw = w = LIMIT;
      n = input_itemsets[(i - 1) * max_cols + j];
    } else {
    }

    // traceback = maximum(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + referrence[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if (traceback == new_nw)
      traceback = nw;
    if (traceback == new_w)
      traceback = w;
    if (traceback == new_n)
      traceback = n;

    fprintf(fpo, "%d ", traceback);

    if (traceback == nw) {
      i--;
      j--;
      continue;
    }

    else if (traceback == w) {
      j--;
      continue;
    }

    else if (traceback == n) {
      i--;
      continue;
    }

    else
      ;
  }

  fclose(fpo);

#endif

  free(referrence);
  free(input_itemsets);
  free(output_itemsets);
}
