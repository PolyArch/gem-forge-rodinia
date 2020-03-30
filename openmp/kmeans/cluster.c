#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmeans.h"

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int numObjects,    /* number of input objects */
            int numAttributes, /* size of attribute of each object */
            float *attributes, /* [numObjects][numAttributes] */
            int nclusters, float threshold, /* in:   */
            float **cluster_centres /* out: [best_nclusters][numAttributes] */

) {

  uint64_t *membership = (uint64_t *)malloc(numObjects * sizeof(uint64_t));

  srand(7);
  /* perform regular Kmeans */
  float *tmp_cluster_centres = kmeans_clustering(
      attributes, numAttributes, numObjects, nclusters, threshold, membership);

  *cluster_centres = tmp_cluster_centres;

  free(membership);

  return 0;
}
