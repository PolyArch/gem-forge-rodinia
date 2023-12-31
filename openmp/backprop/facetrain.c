#include "backprop.h"
#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern char *strcpy();
extern void exit();

int layer_size = 0;
int num_threads = 0;

void backprop_face() {
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  // entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr,
            "usage: backprop <num of input elements> <num of threads>\n");
    exit(0);
  }

  layer_size = atoi(argv[1]);
  num_threads = atoi(argv[2]);
  printf("Use number of threads %d.\n", num_threads);
  int seed;

  seed = 7;
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
