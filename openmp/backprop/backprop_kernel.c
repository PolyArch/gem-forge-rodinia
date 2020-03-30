#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "backprop.h"

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1,
                              int n2);

extern void bpnn_output_error(float *delta, float *target, float *output,
                              int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                              float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                                float **w, float **oldw);

extern int setup(int argc, char **argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { setup(argc, argv); }

void bpnn_train_kernel(BPNN *net, float *eo, float *eh) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  printf("Performing CPU computation\n");
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  omp_set_schedule(omp_sched_static, 0);
#ifdef GEM_FORGE
  mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,
                    hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);
#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
}
