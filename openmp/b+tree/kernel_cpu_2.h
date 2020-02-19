#ifndef KERNEL_CPU_2_H
#define KERNEL_CPU_2_H
void kernel_cpu_2(int cores_arg, knode *knodes, long knodes_elem, int order,
                  long maxheight, int count, long *currKnode, long *offset,
                  long *lastKnode, long *offset_2, int *start, int *end,
                  int *recstart, int *reclength);
#endif