// EXAMPLE:
// ./a.out -file ./input/mil.txt -cores 16
// ...then enter any of the following commands after the prompt > :
// f <x>  -- Find the value under key <x>
// p <x> -- Print the path from the root to key k and its associated value
// t -- Print the B+ tree
// l -- Print the keys of the leaves (bottom row of the tree)
// v -- Toggle output of pointer addresses ("verbose") in tree and leaves.
// k <x> -- Run <x> bundled queries on the CPU and GPU (B+Tree) (Selects random
// values for each search) j <x> <y> -- Run a range search of <x> bundled
// queries on the CPU and GPU (B+Tree) with the range of each search of size <y>
// x <z> -- Run a single search for value z on the GPU and CPU
// y <a> <b> -- Run a single range search for range a-b on the GPU and CPU
// q -- Quit. (Or use Ctl-D.)

#include <limits.h> // (in directory known to compiler)			needed by INT_MIN, INT_MAX
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr
// #include <sys/time.h> //
// (in directory known to compiler)			needed by ???
#include <malloc.h>
#include <math.h> // (in directory known to compiler)			needed by log, pow
#include <omp.h>
#include <string.h> // (in directory known to compiler)			needed by memset

#include "common.h" // (in directory provided here)

#include "num.h"   // (in directory provided here)
#include "timer.h" // (in directory provided here)

#include "impl.h"
#include "kernel_query.h" // (in directory provided here)
#include "kernel_range.h" // (in directory provided here)

int *readInputText(const char *FN) {
  if (FN != NULL) {

    printf("Getting input from file %s...\n", FN);
    // open input file
    FILE *f = fopen(FN, "r");
    if (f == NULL) {
      perror("Failure to open input file.");
      exit(EXIT_FAILURE);
    }

    // get # of numbers in the file, this is global.
    fscanf(f, "%ld\n", &size);
    int *values = malloc(sizeof(int) * size);

    // save all numbers
    for (int i = 0; i < size; ++i) {
      assert(!feof(f) && "Not enough input in file.");
      int input;
      fscanf(f, "%d\n", &input);
      values[i] = input;
    }

    // close file
    fclose(f);
    return values;

  } else {
    printf("ERROR: Argument -file missing\n");
    return NULL;
  }
}

int *readInputFake(const char *FN) {
  // Fake the input.
  size = 1000000;
  int *values = malloc(sizeof(int) * size);
  for (int i = 0; i < size; ++i) {
    values[i] = i;
  }
  return values;
}

node *readInput(const char *FN, bool isFake) {
  int *values = NULL;
  if (!isFake) {
    values = readInputText(FN);
  } else {
    values = readInputFake(FN);
  }
  node *root = NULL;
  for (int i = 0; i < size; ++i) {
    int input = values[i];
    root = insert(root, input, input);
  }
  free(values);
  return root;
}

struct CommonOMPArgs {
  int nthreads;
  knode *knodes;
  long knodes_elem;
  record *records;
  long records_elem;
};

void doQuery(struct CommonOMPArgs args, int count, int *keys, record *ans) {
  kernel_query(args.nthreads, args.records, args.knodes, args.knodes_elem,
               order, maxheight, count, keys, ans);

#ifdef DUMP_OUTPUT
  const char *output = "output.txt";
  FILE *pFile = fopen(output, "aw+");
  if (pFile == NULL) {
    fprintf(stderr, "Fail to open %s !\n", output);
    exit(1);
  }

  fprintf(pFile, "\n ******command: k count=%d \n", count);
  for (int i = 0; i < count; i++) {
    fprintf(pFile, "%d     %d     %d\n", i, keys[i], ans[i].value);
  }
  fprintf(pFile, " \n");
  fclose(pFile);
#endif
}

void doRange(struct CommonOMPArgs args, int count, int rSize, int *start,
             int *end, int *recstart, int *reclength) {
  kernel_range(args.nthreads, args.knodes, args.knodes_elem, order, maxheight,
               count, start, end, recstart, reclength);

#ifdef DUMP_OUTPUT
  const char *output = "output.txt";
  FILE *pFile = fopen(output, "aw+");
  if (pFile == NULL) {
    fprintf(stderr, "Fail to open %s !\n", output);
    exit(1);
  }
  fprintf(pFile, "\n******command: j count=%d, rSize=%d \n", count, rSize);
  for (int i = 0; i < count; i++) {
    fprintf(pFile, "%d    %d    %d    %d\n", i, start[i], recstart[i],
            reclength[i]);
  }
  fprintf(pFile, " \n");
  fclose(pFile);
#endif
}

int main(int argc, char **argv) {
  // assing default values
  int cores_arg = 1;
  char *input_file = NULL;
  bool isFake = false;
  bool isBinary = false;

  // go through arguments
  for (int cur_arg = 1; cur_arg < argc; cur_arg++) {
    if (strcmp(argv[cur_arg], "cores") == 0) {
      // check if value provided
      if (argc >= cur_arg + 1) {
        // check if value is a number
        if (isInteger(argv[cur_arg + 1]) == 1) {
          cores_arg = atoi(argv[cur_arg + 1]);
          if (cores_arg < 0) {
            printf("ERROR: Wrong value to cores parameter, cannot be <=0\n");
            return -1;
          }
          cur_arg = cur_arg + 1;
        } else {
          // value is not a number
          printf("ERROR: Value to cores parameter in not a number\n");
          return 0;
        }
      }
    }
    // check if -file
    else if (strcmp(argv[cur_arg], "file") == 0) {
      // check if value provided
      if (argc >= cur_arg + 1) {
        input_file = argv[cur_arg + 1];
        cur_arg = cur_arg + 1;
        // value is not a number
      }
      // value not provided
      else {
        printf("ERROR: Missing value to -file parameter\n");
        return -1;
      }
    } else if (strcmp(argv[cur_arg], "fake") == 0) {
      isFake = true;
    } else if (strcmp(argv[cur_arg], "binary") == 0) {
      isBinary = true;
    }
  }
  // ------------------------------------------------------------60
  // get input from file, if file provided
  // ------------------------------------------------------------60

  long knodes_elem = 0;
  if (!isBinary) {
    // Normall read in the data and transform to cuda.
    node *root = readInput(input_file, isFake);
    printf("Transforming data to a GPU suitable structure...\n");
    knodes_elem = transform_to_cuda(root, 0);
    maxheight = height(root);
    // Dump to file.
    FILE *o = fopen("btree.data", "wb");
    fwrite(&size, sizeof(size), 1, o);
    fwrite(&maxheight, sizeof(maxheight), 1, o);
    fwrite(&order, sizeof(order), 1, o);
    fwrite(&knodes_elem, sizeof(knodes_elem), 1, o);
    fwrite(knodes, sizeof(knodes[0]), knodes_elem, o);
    fwrite(krecords, sizeof(krecords[0]), size, o);

    fclose(o);
  } else {
    // Read in the data.
    FILE *o = fopen(input_file, "rb");
    fread(&size, sizeof(size), 1, o);
    fread(&maxheight, sizeof(maxheight), 1, o);
    fread(&order, sizeof(order), 1, o);
    fread(&knodes_elem, sizeof(knodes_elem), 1, o);
    knodes = malloc(sizeof(knode) * knodes_elem);
    fread(knodes, sizeof(knodes[0]), knodes_elem, o);
    krecords = malloc(sizeof(krecords[0]) * size);
    fread(krecords, sizeof(krecords[0]), size, o);

    fclose(o);
  }
  printf("knodes %ld.\n", knodes_elem);

  // kernel query range 10000
  int query_count = 10000;
  printf("******command: k count=%d \n", query_count);
  if (query_count > 65535) {
    printf("ERROR: Number of requested querries should be 65,535 at most. "
           "(limited by # of CUDA blocks)\n");
    exit(0);
  }
  int *keys = (int *)malloc(query_count * sizeof(int));
  for (int i = 0; i < query_count; i++) {
    keys[i] = (rand() / (float)RAND_MAX) * size;
  }
  record *ans = (record *)malloc(sizeof(record) * query_count);
  for (int i = 0; i < query_count; i++) {
    ans[i].value = -1;
  }

  // kernel_range
  int range_count = 6000;
  int range_rSize = 3000;
  if (range_rSize > size || range_rSize < 0) {
    printf("Search range size is larger than data set size %d.\n", (int)size);
    exit(0);
  }

  int *start = (int *)malloc(range_count * sizeof(int));
  int *end = (int *)malloc(range_count * sizeof(int));
  // INPUT: start, end CPU initialization
  for (int i = 0; i < range_count; i++) {
    start[i] = (rand() / (float)RAND_MAX) * size;
    end[i] = start[i] + range_rSize;
    if (end[i] >= size) {
      start[i] = start[i] - (end[i] - size);
      end[i] = size - 1;
    }
  }

  int *recstart = (int *)malloc(range_count * sizeof(int));
  int *reclength = (int *)malloc(range_count * sizeof(int));
  for (int i = 0; i < range_count; i++) {
    recstart[i] = 0;
    reclength[i] = 0;
  }

  omp_set_num_threads(cores_arg);
#ifdef GEM_FORGE
  // mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

  printf("nthreads = %d.\n", cores_arg);

#ifdef GEM_FORGE
  m5_detail_sim_start();
#ifdef GEM_FORGE_WARM_CACHE
  for (int i = 0; i < size * sizeof(record); i += 64) {
    volatile char c = ((char *)krecords)[i];
  }
  for (int i = 0; i < knodes_elem * sizeof(knode); i += 64) {
    volatile char c = ((char *)knodes)[i];
  }
// Just do a dummy omp parallel loop to initialize threads.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < cores_arg; ++i) {
    volatile char c = ((char *)knodes)[i];
  }
  printf("Warm up done.");
  m5_reset_stats(0, 0);
#endif
#endif
  struct CommonOMPArgs args;
  args.nthreads = cores_arg;
  args.records = krecords;
  args.records_elem = size;
  args.knodes = knodes;
  args.knodes_elem = knodes_elem;
  doQuery(args, query_count, keys, ans);
  doRange(args, range_count, range_rSize, start, end, recstart, reclength);

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif

  // ------------------------------------------------------------60
  // free remaining memory and exit
  // ------------------------------------------------------------60

  free(start);
  free(end);
  free(recstart);
  free(reclength);
  free(keys);
  free(ans);
  free(krecords);
  free(knodes);
  return EXIT_SUCCESS;
}
