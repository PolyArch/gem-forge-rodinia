#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <malloc.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#define MAX_ARGS 10
#define REC_LENGTH 49     // size of a record in db
#define REC_WINDOW 2097152 // number of records to read at a time
#define LATITUDE_POS 28   // location of latitude coordinates in input record
#define OPEN 10000        // initial value of nearest neighbors
struct Neighbor {
  char entry[REC_LENGTH];
  uint32_t update_idx;
  double dist;
};

struct __attribute__((packed)) Record {
  int year;
  int month;
  int date;
  int hour;
  int num;
  int speed;
  int press;
  float lat;
  float lon;
  char name[10];
};

/**
 * This program finds the k-nearest neighbors
 * Usage:	./nn <filelist> <num> <target latitude> <target longitude>
 *             <threads>
 * filelist: File with the filenames to the records
 * num: Number of nearest neighbors to find
 * target lat: Latitude coordinate for distance calculations
 * target long: Longitude coordinate for distance calculations
 * binary: whether the input is binary format.
 * threads: number of threads.
 *
 * The filelist and data are generated by hurricane_gen.c
 * REC_WINDOW has been arbitrarily assigned; A larger value would allow more
 * work for the threads
 */

#ifndef IS_BINARY
#define IS_BINARY 1
#endif

#if IS_BINARY == 1
struct Record sandbox[REC_WINDOW];
#else
char sandbox[REC_LENGTH * REC_WINDOW];
#endif
int64_t rec_count;
void final_reduce(int k, int threads, struct Neighbor *neighbors,
                  struct Neighbor *reduce_neighbors) {
  for (int i = 0; i < threads * k; i++) {
    float dist = reduce_neighbors[i].dist;
    uint32_t update_idx = reduce_neighbors[i].update_idx;
    reduce_neighbors[i].update_idx = UINT32_MAX;
    if (update_idx != UINT32_MAX) {
      float max_dist = -1;
      int max_idx = 0;
      // find a neighbor with greatest dist and take his spot if allowed!
      for (int j = 0; j < k; j++) {
        double dist = neighbors[j].dist;
        if (dist > max_dist) {
          max_dist = dist;
          max_idx = j;
        }
      }
      // compare each record with max value to find the nearest neighbor
      if (dist < max_dist) {
        neighbors[max_idx].dist = dist;
        neighbors[max_idx].update_idx = update_idx;
      }
    }
  }

  // Set the final value.
  for (int i = 0; i < k; ++i) {
    uint32_t update_idx = neighbors[i].update_idx;
    neighbors[i].update_idx = UINT32_MAX;
    if (update_idx != UINT32_MAX) {
#if IS_BINARY == 1
      strcpy(neighbors[i].entry, sandbox[update_idx].name);
#else
      sandbox[(update_idx + 1) * REC_LENGTH - 1] = '\0';
      strcpy(neighbors[i].entry, sandbox + update_idx * REC_LENGTH);
#endif
    }
  }
}

void process(FILE *flist, struct Neighbor *neighbors,
             struct Neighbor *reduce_neighbors, int k, int threads,
             float target_lat, float target_long) {

  char dbname[64];
  if (fscanf(flist, "%s\n", dbname) != 1) {
    fprintf(stderr, "error reading filelist\n");
    exit(0);
  }

  FILE *fp = fopen(dbname, "r");
  if (!fp) {
    printf("error opening flist\n");
    exit(1);
  }

  double *z = (double *)malloc(REC_WINDOW * sizeof(double));

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
#define VOLATILE_LOAD(x, i) volatile uint8_t x##V = ((uint8_t *)x)[i];
  for (uint64_t i = 0; i < k * sizeof(struct Neighbor); i += 64) {
    VOLATILE_LOAD(neighbors, i);
  }
  for (uint64_t i = 0; i < threads * k * sizeof(struct Neighbor); i += 64) {
    VOLATILE_LOAD(reduce_neighbors, i);
  }
  for (uint64_t i = 0; i < REC_WINDOW * sizeof(float); i += 64) {
    VOLATILE_LOAD(z, i);
  }
// Make sure we start the threads before entering ROI.
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < threads; ++i) {
    VOLATILE_LOAD(reduce_neighbors, i);
  }
#undef VOLATILE_LOAD
  m5_reset_stats(0, 0);
#endif
#endif

  int done = 0;
  while (!done) {

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

// Read in REC_WINDOW number of records
#if IS_BINARY == 1
    rec_count = fread(sandbox, sizeof(struct Record), REC_WINDOW, fp);
#else
    rec_count = fread(sandbox, REC_LENGTH, REC_WINDOW, fp);
#endif
    if (rec_count != REC_WINDOW) {
      if (!ferror(flist)) { // an eof occured
        fclose(fp);

        if (feof(flist))
          done = 1;
        else {
          if (fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
          }

          fp = fopen(dbname, "r");

          if (!fp) {
            printf("error opening a db\n");
            exit(1);
          }
        }
      } else {
        perror("Error");
        exit(0);
      }
    }

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

#pragma omp parallel for firstprivate(z, target_lat, target_long, rec_count)   \
    schedule(static)
    for (int64_t i = 0; i < rec_count; i++) {
#if IS_BINARY == 1
      float tmp_lat = sandbox[i].lat;
      float tmp_long = sandbox[i].lon;
#else
      char *rec_iter = sandbox + (i * REC_LENGTH + LATITUDE_POS - 1);
      float tmp_lat = atof(rec_iter);
      float tmp_long = atof(rec_iter + 5);
#endif
      z[i] = (((tmp_lat - target_lat) * (tmp_lat - target_lat)) +
              ((tmp_long - target_long) * (tmp_long - target_long)));
    }

#ifdef GEM_FORGE
    m5_work_end(1, 0);
    m5_work_begin(2, 0);
#endif

#pragma omp parallel for firstprivate(rec_count, z, reduce_neighbors)          \
    schedule(static)
    for (int i = 0; i < rec_count; i++) {
      int tid = omp_get_thread_num();
      double max_dist = -1;
      int max_idx = 0;
      // find a neighbor with greatest dist and take his spot if allowed!
      int reduce_start = tid * k;
      for (int j = reduce_start; j < reduce_start + k; j++) {
        double dist = reduce_neighbors[j].dist;
        if (dist > max_dist) {
          max_dist = dist;
          max_idx = j;
        }
      }
      // compare each record with max value to find the nearest neighbor
      double dist = z[i];
      if (dist < max_dist) {
        reduce_neighbors[max_idx].dist = dist;
        reduce_neighbors[max_idx].update_idx = i;
      }
    }

#ifdef GEM_FORGE
    m5_work_end(2, 0);
    m5_work_begin(3, 0);
#endif

    // Final reduce.
    final_reduce(k, threads, neighbors, reduce_neighbors);
#ifdef GEM_FORGE
    m5_work_end(3, 0);
#endif
  } // End while loop

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
}

int main(int argc, char *argv[]) {
  long long time0 = clock();

  if (argc < 6) {
    fprintf(stderr, "Invalid set of arguments\n");
    exit(-1);
  }

  FILE *flist = fopen(argv[1], "r");
  if (!flist) {
    printf("error opening flist\n");
    exit(1);
  }

  int k = atoi(argv[2]);
  float target_lat = atof(argv[3]);
  float target_long = atof(argv[4]);
  int threads = atoi(argv[5]);
  omp_set_num_threads(threads);
#ifdef GEM_FORGE
  mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

  struct Neighbor *neighbors = malloc(k * sizeof(struct Neighbor));
  struct Neighbor *reduce_neighbors =
      malloc(threads * k * sizeof(struct Neighbor));
  if (neighbors == NULL || reduce_neighbors == NULL) {
    fprintf(stderr, "no room for neighbors\n");
    exit(1);
  }
  for (int j = 0; j < k; j++) {
    // Initialize list of nearest neighbors to very large dist
    neighbors[j].dist = OPEN;
    neighbors[j].update_idx = UINT32_MAX;
  }
  for (int j = 0; j < k * threads; j++) {
    // Initialize list of nearest neighbors to very large dist
    reduce_neighbors[j].dist = OPEN;
    reduce_neighbors[j].update_idx = UINT32_MAX;
  }

  /**** main processing ****/
  process(flist, neighbors, reduce_neighbors, k, threads, target_lat,
          target_long);

  fprintf(stderr, "The %d nearest neighbors are:\n", k);
  for (int j = 0; j < k; j++) {
    if (!(neighbors[j].dist == OPEN))
      fprintf(stderr, "%s --> %f\n", neighbors[j].entry, neighbors[j].dist);
  }

  fclose(flist);

  long long time1 = clock();
  printf("total time : %15.12fs\n", (float)(time1 - time0) / 1000000);
  return 0;
}
