
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"       // (in directory provided here)
#include "kernel_range.h" // (in directory provided here)
#include "timer.h"        // (in directory provided here)	needed by timer

void kernel_range(int cores_arg, knode *knodes, long knodes_elem, int order,
                  long maxheight, int count, int *start, int *end,
                  int *recstart, int *reclength) {

  //======================================================================================================================================================150
  //	Variables
  //======================================================================================================================================================150

  omp_set_num_threads(cores_arg);
  printf("OMP threads = %d\n", cores_arg);

#ifdef DUMP_OUTPUT
  long long time1 = get_time();
#endif

#ifdef GEM_FORGE
  m5_work_begin(1, 0);
#endif

// process number of querries
#pragma omp parallel for firstprivate(order) schedule(static)
  for (uint64_t bid = 0; bid < count; bid++) {

    const int targetStart = start[bid];
    const int targetEnd = end[bid];
    knode *lhsKnode = &knodes[0];
    knode *rhsKnode = &knodes[0];
    knode *nextLHSKnode = NULL;
    knode *nextRHSKnode = NULL;

    // process levels of the tree
    for (uint64_t i = 0; i < maxheight; i++) {

      // process all leaves at each level
      for (uint64_t thid = 0; thid < order; thid++) {

        int lhsKey1 = lhsKnode->keys[thid];
        int lhsKey2 = lhsKnode->keys[thid + 1];
        if (lhsKey1 <= targetStart && lhsKey2 > targetStart) {
          // this conditional statement is inserted to avoid crush due to but in
          // original code "offset[bid]" calculated below that later addresses
          // part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main
          // function are out of bounds of knodes that they address
          int indice = lhsKnode->indices[thid];
          if (indice < knodes_elem) {
            nextLHSKnode = &knodes[indice];
          }
        }
        int rhsKey1 = rhsKnode->keys[thid];
        int rhsKey2 = rhsKnode->keys[thid + 1];
        if (rhsKey1 <= targetEnd && rhsKey2 > targetEnd) {
          int indice = rhsKnode->indices[thid];
          if (indice < knodes_elem) {
            nextRHSKnode = &knodes[indice];
          } else {
            assert(false);
          }
        }
      }

      assert(nextLHSKnode);
      lhsKnode = nextLHSKnode;
      assert(nextRHSKnode);
      rhsKnode = nextRHSKnode;
    }

    // process leaves
    int startRec = 0;
    for (uint64_t thid = 0; thid < order; thid++) {
      // Find the index of the starting record
      if (lhsKnode->keys[thid] == targetStart) {
        startRec = lhsKnode->indices[thid];
        break;
      }
    }

    // process leaves
    int endRec = 0;
    for (uint64_t thid = 0; thid < order; thid++) {
      // Find the index of the ending record
      if (rhsKnode->keys[thid] == targetEnd) {
        endRec = rhsKnode->indices[thid];
        break;
      }
    }
    recstart[bid] = startRec;
    reclength[bid] = endRec - startRec + 1;
  }

#ifdef GEM_FORGE
  m5_work_end(1, 0);
#endif

#ifdef DUMP_OUTPUT
  long long time2 = get_time();
  printf("Time spent in different stages of CPU/MCPU KERNEL:\n");
  printf("Total time:\n");
  printf("%.12f s\n", (float)(time2 - time1) / 1000000);
#endif
}