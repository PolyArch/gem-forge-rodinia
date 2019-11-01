/**
  It is so slow to read in the txt file in gem5.
  Let's convert them to float binary file.
  Assume little endien.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char o_fn[1024];

int main(int argc, char *argv[]) {
  char *fn = argv[1];
  FILE *infile;
  char line[1024];

  if ((infile = fopen(fn, "r")) == NULL) {
    fprintf(stderr, "Error: no such file (%s)\n", fn);
    exit(1);
  }

  int32_t num_of_nodes;
  fscanf(infile, "%d", &num_of_nodes);
  // Read in the pair of start, edgeno.
  uint32_t *start_edge_no = malloc(sizeof(uint32_t) * num_of_nodes * 2);
  for (unsigned int i = 0; i < num_of_nodes; i++) {
    fscanf(infile, "%d %d", &start_edge_no[i * 2 + 0],
           &start_edge_no[i * 2 + 1]);
  }
  // Read in the source.
  int32_t source;
  fscanf(infile, "%d", &source);

  int32_t edge_list_size;
  fscanf(infile, "%d", &edge_list_size);
  uint32_t *edge_cost = malloc(sizeof(uint32_t) * edge_list_size * 2);
  for (int i = 0; i < edge_list_size; ++i) {
    fscanf(infile, "%d", &edge_cost[i * 2 + 0]); // id
    fscanf(infile, "%d", &edge_cost[i * 2 + 1]); // cost
  }

  fclose(infile);

  o_fn[0] = 0;
  strcat(o_fn, fn);
  strcat(o_fn, ".data");

  FILE *o = fopen(o_fn, "wb");
  fwrite(&num_of_nodes, sizeof(num_of_nodes), 1, o);
  fwrite(start_edge_no, sizeof(start_edge_no[0]), num_of_nodes * 2, o);
  fwrite(&source, sizeof(source), 1, o);
  fwrite(&edge_list_size, sizeof(edge_list_size), 1, o);
  fwrite(edge_cost, sizeof(edge_cost[0]), edge_list_size * 2, o);

  fclose(o);

  return 0;
}
