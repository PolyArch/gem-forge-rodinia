#ifndef KERNEL_CPU_H
#define KERNEL_CPU_H
void kernel_cpu(int cores_arg, record *records, knode *knodes, long knodes_elem,
                int order, long maxheight, int count, long *currKnode,
                long *offset, int *keys, record *ans);
#endif