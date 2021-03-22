
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

void read(float *dest, int dim, int num) {
  for (int i = 0; i < num; i++) {
    for (int k = 0; k < dim; k++) {
      dest[i * dim + k] = lrand48() / (float)INT_MAX;
    }
  }
}

int main(int argc, char *argv[]) {
    int dim = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *fn = argv[3];
    
    FILE *f = fopen(fn, "wb");
    float *dest = malloc(sizeof(float) * dim * N);
    read(dest, dim, N);
    fwrite(dest, sizeof(float), dim * N, f);
    fclose(f);

    return 0;
}