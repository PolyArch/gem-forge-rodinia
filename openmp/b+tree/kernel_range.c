
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

    {
      // Peel down the iteration for the root node.
      int nextLHSKnodeId = -1;
      int nextRHSKnodeId = -1;
      for (uint64_t thid = 0; thid < order; thid++) {

        int lhsKey1 = lhsKnode->keys[thid];
        int lhsKey2 = lhsKnode->keys[thid + 1];
        if (lhsKey1 <= targetStart && lhsKey2 > targetStart) {
          nextLHSKnodeId = thid;
        }
        if (lhsKey1 <= targetEnd && lhsKey2 > targetEnd) {
          nextRHSKnodeId = thid;
        }
      }

      lhsKnode = &knodes[lhsKnode->indices[nextLHSKnodeId]];
      rhsKnode = &knodes[rhsKnode->indices[nextRHSKnodeId]];
    }

    // process remaining levels of the tree
    for (uint64_t i = 1; i < maxheight; i++) {
      // process all leaves at each level
      int nextLHSKnodeId = -1;
      int nextRHSKnodeId = -1;
      for (uint64_t thid = 0; thid < order; thid++) {

        int lhsKey1 = lhsKnode->keys[thid];
        int lhsKey2 = lhsKnode->keys[thid + 1];
        if (lhsKey1 <= targetStart && lhsKey2 > targetStart) {
          nextLHSKnodeId = thid;
        }
        int rhsKey1 = rhsKnode->keys[thid];
        int rhsKey2 = rhsKnode->keys[thid + 1];
        if (rhsKey1 <= targetEnd && rhsKey2 > targetEnd) {
          nextRHSKnodeId = thid;
        }
      }

      lhsKnode = &knodes[lhsKnode->indices[nextLHSKnodeId]];
      rhsKnode = &knodes[rhsKnode->indices[nextRHSKnodeId]];
    }
     
    


    // process leaves
    uint64_t startRecId = UINT64_MAX;
    uint64_t endRecId = UINT64_MAX;
    for (uint64_t thid = 0; thid < order; thid++) {
      // Find the index of the starting record
      if (lhsKnode->keys[thid] == targetStart) {
        startRecId = thid;
      }
      if (rhsKnode->keys[thid] == targetEnd) {
        endRecId = thid;
      }
    }

    int startRec = startRecId != UINT64_MAX ? lhsKnode->indices[startRecId] : 0;
    int endRec = endRecId != UINT64_MAX ? rhsKnode->indices[endRecId] : 0;
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