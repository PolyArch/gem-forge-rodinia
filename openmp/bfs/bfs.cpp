#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

// Structure to hold a node information
struct Node {
  int starting;
  int no_of_edges;
};

void Usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <num_threads> <input_file>\n", argv[0]);
}

void bfs(int nNodes, int nEdges, Node *nodes, int *edges, int *costs,
         bool *masks, bool *updates, bool *visits) {

  int k = 0;
  bool unfinished;
  do {
    // if no thread changes this value then the loop stops
    printf("Begin k = %d.\n", k);

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif
#pragma omp parallel for firstprivate(nNodes, masks, nodes, edges, visits,     \
                                      costs, updates, k) schedule(static)
    for (uint64_t tid = 0; tid < nNodes; tid++) {
      uint64_t start = nodes[tid].starting;
      uint64_t end = nodes[tid].no_of_edges + start;
      bool mask = masks[tid];
      masks[tid] = false;
      if (start < end && mask) {
#pragma clang loop vectorize(disable) unroll(disable)
        for (uint64_t i = start; i < end; i++) {
          uint64_t id = edges[i];
          if (!visits[id]) {
            // printf("Update k %d src %lu dst %lu.\n", k, tid, id);
            costs[id] = k + 1;
            updates[id] = true;
          }
        }
      }
    }

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

    unfinished = false;
#pragma omp parallel for firstprivate(nNodes, masks, visits, updates)          \
    schedule(static) reduction(||                                              \
                               : unfinished)
    for (uint64_t tid = 0; tid < nNodes; tid++) {
      bool update = updates[tid];
      unfinished = unfinished || update;
      updates[tid] = false;
      if (update) {
        masks[tid] = true;
        visits[tid] = true;
      }
    }

#ifdef GEM_FORGE
    m5_work_end(1, 0);
#endif
    k++;
  } while (unfinished);
}

int main(int argc, char **argv) {

  if (argc != 3) {
    Usage(argc, argv);
    exit(0);
  }

  int num_omp_threads = atoi(argv[1]);
  char *input_f = argv[2];

  printf("Reading File\n");
  // Read in Graph from a file
  FILE *fp = fopen(input_f, "rb");
  if (!fp) {
    printf("Error Reading graph file\n");
    return 1;
  }

  int nNodes = 0;
  fread(&nNodes, sizeof(nNodes), 1, fp);

  // allocate host memory
  Node *nodes = (Node *)malloc(sizeof(Node) * nNodes);
  bool *masks = (bool *)malloc(sizeof(bool) * nNodes);
  bool *updates = (bool *)malloc(sizeof(bool) * nNodes);
  bool *visits = (bool *)malloc(sizeof(bool) * nNodes);

  int start, edgeno;
  // initalize the memory
  uint32_t *start_edge_no = (uint32_t *)malloc(sizeof(uint32_t) * nNodes * 2);
  fread(start_edge_no, sizeof(start_edge_no[0]), nNodes * 2, fp);
  for (unsigned int i = 0; i < nNodes; i++) {
    nodes[i].starting = start_edge_no[i * 2 + 0];
    nodes[i].no_of_edges = start_edge_no[i * 2 + 1];
    masks[i] = false;
    updates[i] = false;
    visits[i] = false;
  }
  free(start_edge_no);

  // read the source node from the file
  int source = 0;
  fread(&source, sizeof(source), 1, fp);

  // set the source node as true in the mask
  masks[source] = true;
  visits[source] = true;

  int nEdges = 0;
  fread(&nEdges, sizeof(nEdges), 1, fp);

  int id, cost;
  int *edges = (int *)malloc(sizeof(int) * nEdges);

  uint32_t *edge_cost = (uint32_t *)malloc(sizeof(uint32_t) * nEdges * 2);
  fread(edge_cost, sizeof(edge_cost[0]), nEdges * 2, fp);
  for (int i = 0; i < nEdges; i++) {
    id = edge_cost[i * 2 + 0];
    edges[i] = id;
  }
  free(edge_cost);

  fclose(fp);

  // allocate mem for the result on host side
  int *costs = (int *)malloc(sizeof(int) * nNodes);
  for (int i = 0; i < nNodes; i++)
    costs[i] = -1;
  costs[source] = 0;

  omp_set_num_threads(num_omp_threads);
  kmp_set_stacksize_s(8*1024*1024);
#ifdef GEM_FORGE
  // mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

// ROI Begins.
#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  // 3 masks: visited, mask, updateing.
  size_t mask_size = nNodes * sizeof(masks[0]) * 3;
  size_t node_size = nNodes * sizeof(nodes[0]);
  size_t cost_size = nNodes * sizeof(costs[0]);
  size_t edge_size = nEdges * sizeof(edges[0]);
  printf(
      "Masks %lu kB, nodes %lu kB, cost %lu kB, edge %lu kB, total %lu MB.\n",
      mask_size / 1024, node_size / 1024, cost_size / 1024, edge_size / 1024,
      (mask_size + node_size + cost_size + edge_size) / 1024 / 1024);
#pragma omp parallel for firstprivate(nNodes, masks, visits, updates)          \
    schedule(static)
  for (uint64_t tid = 0; tid < nNodes; tid += 64 / sizeof(bool)) {
    volatile bool mask = masks[tid];
    volatile bool visit = visits[tid];
    volatile bool update = updates[tid];
  }
#pragma omp parallel for firstprivate(nNodes, nodes) schedule(static)
  for (uint64_t tid = 0; tid < nNodes; tid += 64 / sizeof(Node)) {
    volatile Node node = nodes[tid];
  }
#pragma omp parallel for firstprivate(nNodes, costs) schedule(static)
  for (uint64_t tid = 0; tid < nNodes; tid += 64 / sizeof(int)) {
    volatile int cost = costs[tid];
  }
#pragma omp parallel for firstprivate(nEdges, edges) schedule(static)
  for (uint64_t eid = 0; eid < nEdges; eid += 64 / sizeof(int)) {
    volatile int edge = edges[eid];
  }
  m5_reset_stats(0, 0);
#endif
#endif

  bfs(nNodes, nEdges, nodes, edges, costs, masks, updates, visits);

// ROI ends.
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

  // Store the result into a file
  FILE *fpo = fopen("result.txt", "w");
  for (int i = 0; i < nNodes; i++)
    fprintf(fpo, "%d) cost:%d\n", i, costs[i]);
  fclose(fpo);
  printf("Result stored in result.txt\n");

  // cleanup memory
  free(nodes);
  free(edges);
  free(masks);
  free(updates);
  free(visits);
  free(costs);

  return 0;
}
