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

  uint32_t dim = atoi(argv[2]);
  uint32_t layer = atoi(argv[3]);
  uint32_t total = dim * dim * layer;
  char line[1024];
  float *buf = malloc(sizeof(float) * total);

  if ((infile = fopen(fn, "r")) == NULL) {
    fprintf(stderr, "Error: no such file (%s)\n", fn);
    exit(1);
  }
  for (int i = 0; i < total; ++i) {
    fscanf(infile, "%f", &buf[i]);
  }
  fclose(infile);

  o_fn[0] = 0;
  strcat(o_fn, fn);
  strcat(o_fn, ".data");

  FILE *o = fopen(o_fn, "wb");
  fwrite(buf, sizeof(buf[0]), total, o);

  fclose(o);

  return 0;
}
