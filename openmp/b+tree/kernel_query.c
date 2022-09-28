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

    // {
    //   // Peel down the iteration for root node.
    //   int nextKnodeId = -1;
    //   for (uint64_t thid = 0; thid < order; thid++) {
    //     int lhsKey = curKnode->keys[thid];
    //     int rhsKey = curKnode->keys[thid + 1];
    //     if (lhsKey <= target && rhsKey > target) {
    //       nextKnodeId = thid;
    //     }
    //   }
    //   curKnode = &knodes[curKnode->indices[nextKnodeId]];
    // }

    for (uint64_t i = 0; i < maxheight; i++) {
      int64_t nodeId = 0;
      while (true) {
        int *key = curKnode->keys + nodeId;
#pragma ss stream_name "rodinia.b+tree.query.lhsKey.ld"
        int lhsKey = key[0];
#pragma ss stream_name "rodinia.b+tree.query.rhsKey.ld"
        int rhsKey = key[1];

        int matched = lhsKey <= target && rhsKey > target;
        if (matched) {
          break;
        }
        nodeId++;
      }
#pragma ss stream_name "rodinia.b+tree.query.indice.ld/inner-dep"
      int nextIndice = curKnode->indices[nodeId];
      curKnode = &knodes[nextIndice];
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    // process all leaves at each level
    int64_t valueId = 0;
    bool found = false;
    while (true) {
#pragma ss stream_name "rodinia.b+tree.query.leaf.ld"
      int key = curKnode->keys[valueId];
      bool matched = key == target;
      bool broken = matched || (valueId == order - 1);
      found = found || matched;
      if (broken) {
        break;
      }
      valueId++;
    }
    if (found) {
      int indice = curKnode->indices[valueId];
      ans[bid].value = records[indice].value;
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