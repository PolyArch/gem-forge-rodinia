#ifndef KERNEL_QUERY_H
#define KERNEL_QUERY_H
void kernel_query(int nthreads, record *records, knode *knodes,
                  long knodes_elem, int order, long maxheight, int count,
                  int *keys, record *ans);
#endif