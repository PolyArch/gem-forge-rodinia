#ifndef KERNEL_CPU_2_H
#define KERNEL_CPU_2_H
void kernel_range(int cores_arg, knode *knodes, long knodes_elem, int order,
                  long maxheight, int count, int *start, int *end,
                  int *recstart, int *reclength);
#endif