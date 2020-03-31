/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP
 */
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <malloc.h>

#ifdef GEM_FORGE
#include "gem5/m5ops.h"
#endif

#define PI 3.1415926535897932
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;
/**
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value
 * > input value
 */
int roundDouble(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}
/**
 * Set values of the 3D array to a newValue if that value is equal to the
 * testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, int *array3D, int *dimX, int *dimY,
           int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}
/**
 * Generates a uniformly distributed random number using the provided seed and
 * GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */
double randu(int *seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((double)M));
}
/**
 * Generates a normally distributed random number using the Box-Muller
 * transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a double representing random number generated using the Box-Muller
 * algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing
 * value for normal random distribution
 */
double randn(int *seed, int index) {
  /*Box-Muller algorithm*/
  double u = randu(seed, index);
  double v = randu(seed, index);
  double cosine = cos(2 * PI * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}
/**
 * Sets values of 3D matrix using randomly generated numbers from a normal
 * distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] =
            array3D[x * *dimY * *dimZ + y * *dimZ + z] +
            (int)(5 * randn(seed, 0));
      }
    }
  }
}
/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      double distance = sqrt(pow((double)(x - radius + 1), 2) +
                             pow((double)(y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
    }
  }
}
/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX,
                   int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      double distance =
          sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error,
                   int *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}
/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int *se, int numOnes, double *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}
/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(int *I, int IszX, int IszY, int Nfr, int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = (int)roundDouble(IszY / 2.0);
  int y0 = (int)roundDouble(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  int *newMatrix = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);
  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
 * Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 -
 * (IK[IND] - 228)^2)/ 100
 * @param I The 3D matrix
 * @param ind The current ind array
 * @param numOnes The length of ind array
 * @return A double representing the sum
 */
double calcLikelihoodSum(int *I, int *ind, int numOnes) {
  double likelihoodSum = 0.0;
  int y;
  for (y = 0; y < numOnes; y++)
    likelihoodSum +=
        (pow((I[ind[y]] - 100), 2) - pow((I[ind[y]] - 228), 2)) / 50.0;
  return likelihoodSum;
}
/**
 * Finds the first element in the CDF that is greater than or equal to the
 * provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the
 * last index
 */
int findIndex(double *CDF, int lengthCDF, double value) {
  uint64_t index = 0;
  for (uint64_t x = 1; x < lengthCDF; x++) {
    double curr = CDF[x];
    if (curr >= value && index == 0) {
      index = x;
    }
  }
  return index;
}
/**
 * Finds the first element in the CDF that is greater than or equal to the
 * provided value and returns that index
 * @note This function uses binary search before switching to sequential search
 * @param CDF The CDF
 * @param beginIndex The index to start searching from
 * @param endIndex The index to stop searching
 * @param value The value to find
 * @return The index of value in the CDF; if value is never found, returns the
 * last index
 * @warning Use at your own risk; not fully tested
 */
int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
  if (endIndex < beginIndex)
    return -1;
  int middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
  /*check the value*/
  if (CDF[middleIndex] >= value) {
    /*check that it's good*/
    if (middleIndex == 0)
      return middleIndex;
    else if (CDF[middleIndex - 1] < value)
      return middleIndex;
    else if (CDF[middleIndex - 1] == value) {
      while (middleIndex > 0 && CDF[middleIndex - 1] == value)
        middleIndex--;
      return middleIndex;
    }
  }
  if (CDF[middleIndex] > value)
    return findIndexBin(CDF, beginIndex, middleIndex + 1, value);
  return findIndexBin(CDF, middleIndex - 1, endIndex, value);
}

typedef struct {
  uint64_t Nparticles;
  int max_size;
  int IszX;
  int IszY;
  int Nfr;
  int *I;
  uint64_t countOnes;
  double *objxy;
  int *ind;
  double *likelihood;
  double *weights;
  double *CDF;
  int *seed;
  double *arrayX;
  double *arrayY;
  double *xj;
  double *yj;
  double *u;
  int k;
} KernelArgs;

#define ExtractArgs(t, v) t v = args.v;

/**
 * Kernel 1: Apply motion model.
 */
__attribute__((noinline)) void applyMotionModel(KernelArgs args) {
  ExtractArgs(double *, arrayX);
  ExtractArgs(double *, arrayY);
  ExtractArgs(uint64_t, Nparticles);
  ExtractArgs(int *, seed);
#ifdef GEM_FORGE
  m5_work_begin(0, 0);
#endif
// apply motion model
// draws sample from motion model (random walk). The only prior information
// is that the object moves 2x as fast as in the y direction
#pragma omp parallel for firstprivate(arrayX, arrayY, Nparticles, seed)        \
    schedule(static)
  for (uint64_t x = 0; x < Nparticles; x++) {
    arrayX[x] += 1 + 5 * randn(seed, x);
    arrayY[x] += -2 + 2 * randn(seed, x);
  }
#ifdef GEM_FORGE
  m5_work_end(0, 0);
#endif

  printf("1 applyMotionModel done.\n");
}

/**
 * Kernel 2: Compute likelihood.
 * Indirect access to compute the likelihood of particles.
 */
__attribute__((noinline)) void computeLikelihood(KernelArgs args) {
  ExtractArgs(double *, arrayX);
  ExtractArgs(double *, arrayY);
  ExtractArgs(double *, objxy);
  ExtractArgs(double *, likelihood);
  ExtractArgs(uint64_t, Nparticles);
  ExtractArgs(int, countOnes);
  ExtractArgs(int, max_size);
  ExtractArgs(int, IszX);
  ExtractArgs(int, IszY);
  ExtractArgs(int, Nfr);
  ExtractArgs(int, k);
  ExtractArgs(int *, seed);
  ExtractArgs(int *, I);
  ExtractArgs(int *, ind);
#ifdef GEM_FORGE
  m5_work_begin(1, 0);
#endif
#pragma omp parallel for firstprivate(Nparticles, countOnes, max_size, IszX,   \
                                      IszY, Nfr, k, likelihood, I, arrayX,     \
                                      arrayY, objxy, ind) schedule(static)
  for (uint64_t x = 0; x < Nparticles; x++) {
    // compute the likelihood: remember our assumption is that you know
    // foreground and the background image intensity distribution.
    // Notice that we consider here a likelihood ratio, instead of
    // p(z|x). It is possible in this case. why? a hometask for you.
    // calc ind

    /**
     * Unroll this will generate a load of <4xdouble>, which breaks
     * the backend as we don't know how to split a StreamLoad.
     * TODO: Remove this after we have a transformation to split a load of
     * <4xdouble>
     * TODO: 2 load of <2xdouble>.
     */
    int xValue = roundDouble(arrayX[x]);
    int yValue = roundDouble(arrayY[x]);
    double likelihoodValue = 0.0;
#pragma clang loop vectorize(disable)
    for (uint64_t y = 0; y < countOnes; y++) {
      int indX = xValue + objxy[y * 2 + 1];
      int indY = yValue + objxy[y * 2];
      int indValue = abs(indX * IszY * Nfr + indY * Nfr + k);
      if (indValue >= max_size) {
        indValue = 0;
      }
      int IValue = I[indValue];
      likelihoodValue +=
          (pow((IValue - 100), 2) - pow((IValue - 228), 2)) / 50.0;
    }
    likelihood[x] = likelihoodValue / ((double)countOnes);
  }
#ifdef GEM_FORGE
  m5_work_end(1, 0);
#endif
  printf("2 computeLikelihood done.\n");
}

__attribute__((noinline)) void updateWeight(KernelArgs args) {
  ExtractArgs(uint64_t, Nparticles);
  ExtractArgs(double *, likelihood);
  ExtractArgs(double *, weights);
  double sumWeights = 0;
#ifdef GEM_FORGE
  m5_work_begin(2, 0);
#endif
#pragma omp parallel for firstprivate(Nparticles, weights, likelihood) reduction(+ : sumWeights)
  for (uint64_t x = 0; x < Nparticles; x++) {
    double weight = weights[x];
    weight *= exp(likelihood[x]);
    sumWeights += weight;
    weights[x] = weight;
  }
#ifdef GEM_FORGE
  m5_work_end(2, 0);
#endif
#ifdef GEM_FORGE
  m5_work_begin(3, 0);
#endif
#pragma omp parallel for firstprivate(Nparticles, weights, sumWeights)
  for (uint64_t x = 0; x < Nparticles; x++) {
    double weight = weights[x];
    weights[x] = weight / sumWeights;
  }
#ifdef GEM_FORGE
  m5_work_end(3, 0);
#endif

  printf("3 updateWeight done.\n");
}

__attribute((noinline)) void averageParticles(KernelArgs args) {
  ExtractArgs(double *, arrayX);
  ExtractArgs(double *, arrayY);
  ExtractArgs(uint64_t, Nparticles);
  ExtractArgs(double *, weights);
  ExtractArgs(int, IszX);
  ExtractArgs(int, IszY);
  double xe = 0;
  double ye = 0;
// estimate the object location by expected values
#ifdef GEM_FORGE
  m5_work_begin(4, 0);
#endif
#pragma omp parallel for firstprivate(Nparticles, arrayX, arrayY, weights) reduction(+ : xe, ye)
  for (uint64_t x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
#ifdef GEM_FORGE
  m5_work_end(4, 0);
#endif
  double distance = sqrt(pow((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                         pow((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
  printf("4 averageParticles done (%lf, %lf), %lf\n", xe, ye, distance);
}

__attribute__((noinline)) void resampleParticles(KernelArgs args) {
  ExtractArgs(uint64_t, Nparticles);
  ExtractArgs(double *, weights);
  ExtractArgs(double *, CDF);
  ExtractArgs(double *, arrayX);
  ExtractArgs(double *, arrayY);
  ExtractArgs(double *, xj);
  ExtractArgs(double *, yj);
  ExtractArgs(double *, u);
  ExtractArgs(int *, seed);

#ifdef GEM_FORGE
  m5_work_begin(5, 0);
#endif
  CDF[0] = weights[0];
  for (uint64_t x = 1; x < Nparticles; x++) {
    CDF[x] = weights[x] + CDF[x - 1];
  }
#ifdef GEM_FORGE
  m5_work_end(5, 0);
#endif

  printf("5 computeCDF done.\n");

#ifdef GEM_FORGE
  m5_work_begin(6, 0);
#endif
  double u1 = (1 / ((double)(Nparticles))) * randu(seed, 0);
#pragma omp parallel for firstprivate(u, u1, Nparticles) schedule(static)
  for (uint64_t x = 0; x < Nparticles; x++) {
    u[x] = u1 + x / ((double)(Nparticles));
  }
#ifdef GEM_FORGE
  m5_work_end(6, 0);
#endif

  printf("6 computeU done.\n");

#ifdef GEM_FORGE
  m5_work_begin(7, 0);
#endif
/**
 * To save simulation time, I have decided to reduce the sample number.
 * The simulation time will be adjusted later when processing the results.
 */
#define REDUCE_RESAMPLE 16
#pragma omp parallel for firstprivate(CDF, Nparticles, xj, yj, u, arrayX,      \
                                      arrayY)
  for (uint64_t j = 0; j < Nparticles; j += REDUCE_RESAMPLE) {
    int i = findIndex(CDF, Nparticles, u[j]);
    double resampledX = arrayX[i];
    double resampledY = arrayY[i];
    for (uint64_t k = 0; k < REDUCE_RESAMPLE; ++k) {
      uint64_t idx = j + k;
      xj[idx] = resampledX;
      yj[idx] = resampledY;
    }
  }
#ifdef GEM_FORGE
  m5_work_end(7, 0);
#endif

  printf("7 resample done.\n");

#ifdef GEM_FORGE
  m5_work_begin(8, 0);
#endif
#pragma omp parallel for firstprivate(weights, Nparticles, arrayX, arrayY, xj, \
                                      yj)
  for (uint64_t x = 0; x < Nparticles; x++) {
    // reassign arrayX and arrayY
    arrayX[x] = xj[x];
    arrayY[x] = yj[x];
    weights[x] = 1 / ((double)(Nparticles));
  }
#ifdef GEM_FORGE
  m5_work_end(8, 0);
#endif

  printf("8 reset done.\n");
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In
 * addition, it references a provided MATLAB function which takes the video, the
 * objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 * @param Nthreads The number of threads
 */
void particleFilter(int *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles, int Nthreads) {

  omp_set_num_threads(Nthreads);
#ifdef GEM_FORGE
  mallopt(M_ARENA_MAX, GEM_FORGE_MALLOC_ARENA_MAX);
#endif
  int max_size = IszX * IszY * Nfr;
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)malloc(diameter * diameter * sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  double *objxy = (double *)malloc(countOnes * 2 * sizeof(double));
  getneighbors(disk, countOnes, objxy, radius);

  // initial weights are all equal (1/Nparticles)
  double *weights = (double *)malloc(sizeof(double) * Nparticles);
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((double)(Nparticles));
  }
  // initial likelihood to 0.0
  double *likelihood = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayX = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayY = (double *)malloc(sizeof(double) * Nparticles);
  double *xj = (double *)malloc(sizeof(double) * Nparticles);
  double *yj = (double *)malloc(sizeof(double) * Nparticles);
  double *CDF = (double *)malloc(sizeof(double) * Nparticles);
  double *u = (double *)malloc(sizeof(double) * Nparticles);
  int *ind = (int *)malloc(sizeof(int) * countOnes * Nparticles);
  for (x = 0; x < Nparticles; x++) {
    arrayX[x] = xe;
    arrayY[x] = ye;
  }

#define TO_KB(x) x >> 10
#define TO_MB(x) x >> 20
#define PRINT_SIZE_KB(x) printf("sizeof %10s = %lukB.\n", #x, TO_KB(x##Size))
#define PRINT_SIZE_MB(x) printf("sizeof %10s = %luMB.\n", #x, TO_MB(x##Size))
  {
    size_t objxySize = countOnes * 2 * sizeof(double);
    size_t weightsSize = Nparticles * sizeof(double);
    size_t likelihoodSize = Nparticles * sizeof(double);
    size_t arrayXSize = Nparticles * sizeof(double);
    size_t arrayYSize = Nparticles * sizeof(double);
    size_t xjSize = Nparticles * sizeof(double);
    size_t yjSize = Nparticles * sizeof(double);
    size_t CDFSize = Nparticles * sizeof(double);
    size_t uSize = Nparticles * sizeof(double);
    size_t indSize = Nparticles * countOnes * sizeof(int);
    size_t totalSize = objxySize + weightsSize + likelihoodSize + arrayXSize +
                       arrayYSize + xjSize + yjSize + CDFSize + uSize + indSize;
    PRINT_SIZE_KB(objxy);
    PRINT_SIZE_KB(weights);
    PRINT_SIZE_KB(likelihood);
    PRINT_SIZE_KB(arrayX);
    PRINT_SIZE_KB(arrayY);
    PRINT_SIZE_KB(xj);
    PRINT_SIZE_KB(yj);
    PRINT_SIZE_KB(CDF);
    PRINT_SIZE_KB(u);
    PRINT_SIZE_KB(ind);
    PRINT_SIZE_MB(total);
  }
#undef PRINT_SIZE_MB
#undef PRINT_SIZE_KB
#undef TO_KB
#undef TO_MB

  // Prepare the arguments to the kernel.
  KernelArgs args;
  args.Nparticles = Nparticles;
  args.max_size = max_size;
  args.IszX = IszX;
  args.IszY = IszY;
  args.Nfr = Nfr;
  args.I = I;
  args.countOnes = countOnes;
  args.objxy = objxy;
  args.ind = ind;
  args.likelihood = likelihood;
  args.weights = weights;
  args.CDF = CDF;
  args.seed = seed;
  args.arrayX = arrayX;
  args.arrayY = arrayY;
  args.xj = xj;
  args.yj = yj;
  args.u = u;

#ifdef GEM_FORGE
  omp_set_dynamic(0);
  omp_set_schedule(omp_sched_static, 0);
  m5_detail_sim_start();

#ifdef GEM_FORGE_WARM_CACHE
#define VOLATILE_LOAD(x, i) volatile uint8_t x##V = ((uint8_t *)x)[i];
  for (uint64_t i = 0; i < Nparticles * sizeof(double); i += 64) {
    VOLATILE_LOAD(weights, i);
    VOLATILE_LOAD(likelihood, i);
    VOLATILE_LOAD(arrayX, i);
    VOLATILE_LOAD(arrayY, i);
    VOLATILE_LOAD(xj, i);
    VOLATILE_LOAD(yj, i);
    VOLATILE_LOAD(CDF, i);
    VOLATILE_LOAD(u, i);
  }
  for (uint64_t i = 0; i < countOnes * 2 * sizeof(double); i += 64) {
    VOLATILE_LOAD(objxy, i);
  }
  // Here we start the threads.
  for (uint64_t i = 0; i < countOnes * Nparticles * sizeof(int); i += 64) {
    VOLATILE_LOAD(ind, i);
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Nthreads; ++i) {
    printf("Start thread %d.\n", i);
  }
#undef VOLATILE_LOAD
  m5_reset_stats(0, 0);
#endif

#endif
  for (int k = 1; k < Nfr; k++) {
    args.k = k;
    applyMotionModel(args);
    computeLikelihood(args);
    updateWeight(args);
    averageParticles(args);
    resampleParticles(args);
  }

#ifdef GEM_FORGE
  m5_detail_sim_end();
  exit(0);
#endif
  free(disk);
  free(objxy);
  free(weights);
  free(likelihood);
  free(xj);
  free(yj);
  free(arrayX);
  free(arrayY);
  free(CDF);
  free(u);
  free(ind);
}
int main(int argc, char *argv[]) {

  char *usage =
      "openmp.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles> -nt <NThreads>";
  // check number of arguments
  if (argc != 11) {
    printf("%s\n", usage);
    return 0;
  }
  // check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") ||
      strcmp(argv[7], "-np") || strcmp(argv[9], "-nt")) {
    printf("%s\n", usage);
    return 0;
  }

  int Nthreads = atoi(argv[10]);

  int IszX, IszY, Nfr, Nparticles;

  // converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }
  if ((Nparticles % REDUCE_RESAMPLE) != 0) {
    printf("Number of particles must be multiple of %d\n", REDUCE_RESAMPLE);
    return 0;
  }
  // establish seed
  int *seed = (int *)malloc(sizeof(int) * Nparticles);
  int i;
  for (i = 0; i < Nparticles; i++) {
    // seed[i] = time(0) * i;
    seed[i] = i;
  }
  // malloc matrix
  int *I = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  // call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);
  // call particle filter
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles, Nthreads);

  free(seed);
  free(I);
  return 0;
}
