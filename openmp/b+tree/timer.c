#include "timer.h"

#include <stdlib.h>
#include <sys/time.h>

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}