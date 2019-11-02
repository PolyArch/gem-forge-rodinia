/**
  It is so slow to read in the txt file in gem5.
  Let's convert them to float binary file.
  Assume little endien.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

char o_fn[1024];

#define NNB 4
#define NDIM 3

int main(int argc, char *argv[]) {
  char *fn = argv[1];
  FILE *infile;

  char line[1024];

  if ((infile = fopen(fn, "r")) == NULL) {
    fprintf(stderr, "Error: no such file (%s)\n", fn);
    exit(1);
  }
  int nel;
  fscanf(infile, "%d", &nel);
  double *areas = (double *)malloc(sizeof(double) * nel);
  int *elements_surrounding_elements = (int *)malloc(sizeof(int) * NNB * nel);
  double *normals = (double *)malloc(sizeof(double) * nel * NNB * NDIM);
  for (int i = 0; i < nel; ++i) {
    fscanf(infile, "%lf", &areas[i]);
    for (int j = 0; j < NNB; ++j) {
      fscanf(infile, "%d", &elements_surrounding_elements[i * NNB + j]);
      if (elements_surrounding_elements[i * NNB + j] < 0) {
        elements_surrounding_elements[i * NNB + j] = -1;
      }
      elements_surrounding_elements[i * NNB + j]--; // It's coming in with
                                                    // Fortran numbering.
      for (int k = 0; k < NDIM; ++k) {
        double *normal_ptr = &normals[(i * NNB + j) * NDIM + k];
        int n = fscanf(infile, "%lf", normal_ptr);
        assert(n == 1);
        *normal_ptr = -(*normal_ptr);
      }
    }
  }
  fclose(infile);

  o_fn[0] = 0;
  strcat(o_fn, fn);
  strcat(o_fn, ".data");

  FILE *o = fopen(o_fn, "wb");
  fwrite(&nel, sizeof(nel), 1, o);
  fwrite(areas, sizeof(areas[0]), nel, o);
  fwrite(elements_surrounding_elements,
         sizeof(elements_surrounding_elements[0]), nel * NNB, o);
  fwrite(normals, sizeof(normals[0]), nel * NNB * NDIM, o);

  fclose(o);

  return 0;
}
