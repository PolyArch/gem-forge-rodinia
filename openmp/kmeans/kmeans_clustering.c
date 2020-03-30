/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**
 * **/
/**   Edited by: Sang-Ha  Lee
 * **/
/**				 University of Virginia
 * **/
/**
 * **/
/**   Description:	No longer supports fuzzy c-means clustering;
 * **/
/**					only regular k-means clustering.
 * **/
/**					Simplified for main functionality:
 * regular k-means	**/
/**					clustering.
 * **/
/**                                                                     **/
/*************************************************************************/

#include "kmeans.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);
extern uint64_t num_omp_threads;

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
inline float euclid_dist_2(float *pt1, float *pt2, uint64_t numdims) {
  float ans = 0.0;

  for (uint64_t i = 0; i < numdims; i++) {
    ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);
  }

  return ans;
}

uint64_t find_nearest_point(float *pt, /* [nfeatures] */
                            uint64_t nfeatures,
                            float *pts, /* [npts][nfeatures] */
                            uint64_t npts) {
  uint64_t index;
  float min_dist = FLT_MAX;

  /* find the cluster center id with min distance to pt */
  for (uint64_t i = 0; i < npts; i++) {
    float *center = pts + (i * nfeatures);
    float dist = euclid_dist_2(pt, center, nfeatures); /* no need square root */
    if (dist < min_dist) {
      min_dist = dist;
      index = i;
    }
  }
  return index;
}

__attribute__((noinline)) void
kmeans_kernel(float *feature, uint64_t nfeatures, uint64_t npoints,
              uint64_t nclusters, float threshold, uint64_t *membership,
              float *clusters, uint64_t *new_centers_len, float *new_centers,
              uint64_t *partial_new_centers_len, float *partial_new_centers) {
  omp_set_num_threads(num_omp_threads);
  omp_set_dynamic(0);
#ifdef GEM_FORGE
  mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif

#ifdef GEM_FORGE
  m5_detail_sim_start();
#endif

  uint64_t loop = 0;
  do {

#ifdef GEM_FORGE
    m5_work_begin(0, 0);
#endif

#pragma omp parallel shared(feature, clusters, membership,                     \
                            partial_new_centers, partial_new_centers_len)
    {
      uint64_t tid = omp_get_thread_num();
      uint64_t *my_partial_new_centers_len =
          partial_new_centers_len + tid * nclusters;
      float *my_partial_new_centers =
          partial_new_centers + tid * nclusters * nfeatures;
#pragma omp for firstprivate(npoints, nclusters, nfeatures) schedule(static)
      for (uint64_t i = 0; i < npoints; i++) {
        float *point = feature + i * nfeatures;
        /* find the index of nestest cluster centers */
        uint64_t index;
        {
          float min_dist = FLT_MAX;
          for (uint64_t i = 0; i < nclusters; i++) {
            float *center = clusters + (i * nfeatures);
            float dist = euclid_dist_2(point, center, nfeatures);
            if (dist < min_dist) {
              min_dist = dist;
              index = i;
            }
          }
        }

        /* assign the membership to object i */
        membership[i] = index;

        /* update new cluster centers : sum of all objects located
               within */
        my_partial_new_centers_len[index]++;
        for (uint64_t j = 0; j < nfeatures; j++) {
          uint64_t partial_new_center_idx = index * nfeatures + j;
          my_partial_new_centers[partial_new_center_idx] += point[j];
        }
      }
    } /* end of #pragma omp parallel */

#ifdef GEM_FORGE
    m5_work_end(0, 0);
    m5_work_begin(1, 0);
#endif

    /* let the main thread perform the array reduction */
    for (uint64_t j = 0; j < num_omp_threads; j++) {
      for (uint64_t i = 0; i < nclusters; i++) {
        uint64_t partial_len_idx = j * nclusters + i;
        uint64_t partial_len = partial_new_centers_len[partial_len_idx];
        partial_new_centers_len[partial_len_idx] = 0;
        uint64_t new_centers_len_v = new_centers_len[i];
        new_centers_len[i] = new_centers_len_v + partial_len;
        for (uint64_t k = 0; k < nfeatures; k++) {
          uint64_t idx = i * nfeatures + k;
          uint64_t partial_idx = j * nclusters * nfeatures + i * nfeatures + k;
          uint64_t partial_new_center = partial_new_centers[partial_idx];
          partial_new_centers[partial_idx] = 0.0;
          uint64_t new_center = new_centers[idx];
          new_centers[idx] = new_center + partial_new_center;
        }
      }
    }

#ifdef GEM_FORGE
    m5_work_end(1, 0);
    m5_work_begin(2, 0);
#endif

    /* replace old cluster centers with new_centers */
    for (uint64_t i = 0; i < nclusters; i++) {
      uint64_t len = new_centers_len[i];
      len = len == 0 ? 1 : len;
      new_centers_len[i] = 0; /* set back to 0 */
      for (uint64_t j = 0; j < nfeatures; j++) {
        uint64_t idx = i * nfeatures + j;
        clusters[idx] = new_centers[idx] / len;
        new_centers[idx] = 0.0; /* set back to 0 */
      }
    }

#ifdef GEM_FORGE
    m5_work_end(2, 0);
#endif

  } while (loop++ < 500);
#ifdef GEM_FORGE
  m5_detail_sim_end();
#endif
}

/*----< kmeans_clustering() >---------------------------------------------*/
float *kmeans_clustering(float *feature, /* in: [npoints][nfeatures] */
                         uint64_t nfeatures, uint64_t npoints,
                         uint64_t nclusters, float threshold,
                         uint64_t *membership) /* out: [npoints] */
{

  /**
   * Clusters is nclusters x nfeatures x sizeof(float).
   */
  float *clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));

  /* pick cluster centers to the first ones */
  for (uint64_t i = 0; i < nclusters; i++) {
    for (uint64_t j = 0; j < nfeatures; j++) {
      uint64_t idx = i * nfeatures + j;
      clusters[idx] = feature[idx];
    }
  }

  for (uint64_t i = 0; i < npoints; i++)
    membership[i] = -1;

  uint64_t *new_centers_len = (uint64_t *)calloc(nclusters, sizeof(uint64_t));
  float *new_centers = (float *)calloc(nclusters * nfeatures, sizeof(float));

  uint64_t *partial_new_centers_len =
      (uint64_t *)calloc(num_omp_threads * nclusters, sizeof(uint64_t));
  float *partial_new_centers = (float *)calloc(
      num_omp_threads * nclusters * nfeatures, sizeof(float **));

  printf("num of threads = %lu\n", num_omp_threads);

  kmeans_kernel(feature, nfeatures, npoints, nclusters, threshold, membership,
                clusters, new_centers_len, new_centers, partial_new_centers_len,
                partial_new_centers);

  free(new_centers);
  free(new_centers_len);
  free(partial_new_centers);
  free(partial_new_centers_len);

  return clusters;
}
