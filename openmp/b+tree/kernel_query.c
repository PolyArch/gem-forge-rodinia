#include <omp.h>   // (in directory known to compiler)			needed by openmp
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr
#include <stdlib.h> // (in directory known to compiler)			needed by malloc

#include "common.h" // (in directory provided here)
#include "kernel_query.h"
#include "timer.h" // (in directory provided here)

void kernel_query(int nthreads, record *records, knode *knodes,
                  long knodes_elem, int order, long maxheight, int count,
                  int *keys, record *ans) {

  //======================================================================================================================================================150
  //	Variables
  //======================================================================================================================================================150

  // timer
  const int threadsPerBlock = order < 1024 ? order : 1024;

#ifdef DUMP_OUTPUT
  long long time1 = get_time();
#endif

#ifdef GEM_FORGE
  m5_work_begin(0, 0);
#endif

// process number of querries
#pragma omp parallel for schedule(static)
  for (uint64_t bid = 0; bid < count; bid++) {
    int target = keys[bid];
    knode *curKnode = &knodes[0];

    {
      // Peel down the iteration for root node.
      int nextKnodeId = -1;
      for (uint64_t thid = 0; thid < threadsPerBlock; thid++) {
        int lhsKey = curKnode->keys[thid];
        int rhsKey = curKnode->keys[thid + 1];
        if (lhsKey <= target && rhsKey > target) {
          nextKnodeId = thid;
        }
      }
      curKnode = &knodes[curKnode->indices[nextKnodeId]];
    }

    for (uint64_t i = 1; i < maxheight; i++) {
      int nextKnodeId = -1;
      for (uint64_t thid = 0; thid < threadsPerBlock; thid++) {
        int lhsKey = curKnode->keys[thid];
        int rhsKey = curKnode->keys[thid + 1];
        if (lhsKey <= target && rhsKey > target) {
          nextKnodeId = thid;
        }
      }
      curKnode = &knodes[curKnode->indices[nextKnodeId]];
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    // process all leaves at each level
    int valueId = -1;
    for (uint64_t thid = 0; thid < threadsPerBlock; thid++) {
      if (curKnode->keys[thid] == target) {
        valueId = thid;
      }
    }
    if (valueId != -1) {
      ans[bid].value = records[curKnode->indices[valueId]].value;
    }
  }

#ifdef GEM_FORGE
  m5_work_end(0, 0);
#endif

#ifdef DUMP_OUTPUT
  long long time2 = get_time();
  printf("Time spent in different stages of CPU/MCPU KERNEL:\n");
  printf("Total time:\n");
  printf("%.12f s\n", (float)(time2 - time1) / 1000000);
#endif
}