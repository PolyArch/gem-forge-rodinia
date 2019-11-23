#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

//#define NUM_THREAD 4
#define OPEN

FILE *fp;

// Structure to hold a node information
struct Node {
  int starting;
  int no_of_edges;
};

void BFSGraph(int argc, char **argv);

void Usage(int argc, char **argv) {

  fprintf(stderr, "Usage: %s <num_threads> <input_file>\n", argv[0]);
}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { BFSGraph(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
  int no_of_nodes = 0;
  int edge_list_size = 0;
  char *input_f;
  int num_omp_threads;

  if (argc != 3) {
    Usage(argc, argv);
    exit(0);
  }

  num_omp_threads = atoi(argv[1]);
  input_f = argv[2];

  printf("Reading File\n");
  // Read in Graph from a file
  fp = fopen(input_f, "rb");
  if (!fp) {
    printf("Error Reading graph file\n");
    return;
  }

  fread(&no_of_nodes, sizeof(no_of_nodes), 1, fp);

  // allocate host memory
  Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
  bool *h_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_graph_visited = (bool *)malloc(sizeof(bool) * no_of_nodes);

  int start, edgeno;
  // initalize the memory
  uint32_t *start_edge_no =
      (uint32_t *)malloc(sizeof(uint32_t) * no_of_nodes * 2);
  fread(start_edge_no, sizeof(start_edge_no[0]), no_of_nodes * 2, fp);
  for (unsigned int i = 0; i < no_of_nodes; i++) {
    h_graph_nodes[i].starting = start_edge_no[i * 2 + 0];
    h_graph_nodes[i].no_of_edges = start_edge_no[i * 2 + 1];
    h_graph_mask[i] = false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i] = false;
  }
  free(start_edge_no);

  // read the source node from the file
  int source = 0;
  fread(&source, sizeof(source), 1, fp);

  // set the source node as true in the mask
  h_graph_mask[source] = true;
  h_graph_visited[source] = true;

  fread(&edge_list_size, sizeof(edge_list_size), 1, fp);

  int id, cost;
  int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);

  uint32_t *edge_cost =
      (uint32_t *)malloc(sizeof(uint32_t) * edge_list_size * 2);
  fread(edge_cost, sizeof(edge_cost[0]), edge_list_size * 2, fp);
  for (int i = 0; i < edge_list_size; i++) {
    id = edge_cost[i * 2 + 0];
    h_graph_edges[i] = id;
  }
  free(edge_cost);

  if (fp)
    fclose(fp);

  // allocate mem for the result on host side
  int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++)
    h_cost[i] = -1;
  h_cost[source] = 0;

  printf("Start traversing the tree.\n");

  int k = 0;
#ifdef OPEN
  double start_time = omp_get_wtime();
  omp_set_dynamic(0);
  omp_set_num_threads(num_omp_threads);
#ifdef OMP_OFFLOAD
#pragma omp target data map(                                                   \
    to                                                                         \
    : no_of_nodes, h_graph_mask [0:no_of_nodes],                               \
      h_graph_nodes [0:no_of_nodes], h_graph_edges [0:edge_list_size],         \
      h_graph_visited [0:no_of_nodes], h_updating_graph_mask [0:no_of_nodes])  \
    map(h_cost [0:no_of_nodes])
  {
#endif
#endif

// ROI Begins.
#ifdef GEM_FORGE
    m5_detail_sim_start();
#endif

    bool unfinished;
    do {
      // if no thread changes this value then the loop stops
      unfinished = false;

#ifdef OPEN
#ifdef OMP_OFFLOAD
#pragma omp target
#endif
#pragma omp parallel for firstprivate(                                         \
    no_of_nodes, h_graph_mask, h_graph_nodes, h_graph_edges, h_graph_visited,  \
    h_cost, h_updating_graph_mask)
#endif
      for (uint64_t tid = 0; tid < no_of_nodes; tid++) {
        if (h_graph_mask[tid]) {
          h_graph_mask[tid] = false;
          uint64_t start = h_graph_nodes[tid].starting;
          uint64_t end = h_graph_nodes[tid].no_of_edges + start;
          for (uint64_t i = start; i < end; i++) {
            uint64_t id = h_graph_edges[i];
            if (!h_graph_visited[id]) {
              h_cost[id] = h_cost[tid] + 1;
              h_updating_graph_mask[id] = true;
            }
          }
        }
      }

#ifdef OPEN
#ifdef OMP_OFFLOAD
#pragma omp target map(stop)
#endif
#pragma omp parallel for firstprivate(no_of_nodes, h_updating_graph_mask,      \
                                      h_graph_mask, h_graph_visited,           \
                                      h_updating_graph_mask)
#endif
      for (uint64_t tid = 0; tid < no_of_nodes; tid++) {
        if (h_updating_graph_mask[tid]) {
          h_graph_mask[tid] = true;
          h_graph_visited[tid] = true;
          unfinished = true;
          h_updating_graph_mask[tid] = false;
        }
      }
      k++;
    } while (unfinished);

// ROI ends.
#ifdef GEM_FORGE
    m5_detail_sim_end();
    exit(0);
#endif

#ifdef OPEN
    double end_time = omp_get_wtime();
    printf("Compute time: %lf\n", (end_time - start_time));
#ifdef OMP_OFFLOAD
  }
#endif
#endif
  // Store the result into a file
  FILE *fpo = fopen("result.txt", "w");
  for (int i = 0; i < no_of_nodes; i++)
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  fclose(fpo);
  printf("Result stored in result.txt\n");

  // cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
}
