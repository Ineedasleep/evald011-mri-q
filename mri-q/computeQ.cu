/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include "computeQ.cc"
//#include "parboil.h"

#define MU_THREADS_PER_BLOCK 4096
#define Q_THREADS_PER_BLOCK 256

/************ Definitions from computeQ.cc ************
#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};
*/

/* Original
inline
void 
ComputePhiMagCPU(int numK, 
                 float* phiR, float* phiI,
                 float* __restrict__ phiMag) {
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}
*/

__global__
void dev_ComputePhiMagGPU(const int numK, const float* phiR, const float* phiI, float* phiMag) {
    int indexK = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float real = phiR[indexK];
        float imag = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
}

void ComputePhiMagGPU(int numK, float* phiR_d, float* &phiI_d, float* &phiMag_d) {
  dim3 dim_grid((numK-1)/MU_THREADS_PER_BLOCK + 1, 1, 1);
  dim3 dim_block(MU_THREADS_PER_BLOCK, 1, 1);

  dev_ComputePhiMagGPU<<<dim_grid, dim_block>>>(numK, phiR_d, phiI_d, phiMag_d);
}

/* Original
inline
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *__restrict__ Qr, float *__restrict__ Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;

  // Loop over the space and frequency domains.
  // Generally, numX > numK.
  // Since loops are not tiled, it's better that the loop with the smaller
  // cache footprint be innermost.
  for (indexX = 0; indexX < numX; indexX++) {

    // Sum the contributions to this point over all frequencies
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    for (indexK = 0; indexK < numK; indexK++) {
      expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                       kVals[indexK].Ky * y[indexX] +
                       kVals[indexK].Kz * z[indexX]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[indexX] = Qracc;
    Qi[indexX] = Qiacc;
  }
}
*/

/* Original
void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
*/

void createDataStructsGPU(int numK, int numX, 
                          float* &x, float* &y, float* &z,
                          float* &phiR, float* &phiI,
                          float* &x_d, float* &y_d, float* &z_d,
                          float* &phiR_d, float* &phiI_d, float* &phiMag_d,
                          float* &Qr_d, float* &Qi_d) {
  cudaError_t cuda_ret;

  cuda_ret = cudaMalloc((void**) &x_d, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &y_d, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &z_d, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &phiR_d, numK*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &phiI_d, numK*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &phiMag_d, numK*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &Qi_d, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMalloc((void**) &Qr_d, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();

  cuda_ret = cudaMemcpy(x_d, x, numX*sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMemcpy(y_d, y, numX*sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMemcpy(z_d, z, numX*sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMemcpy(phiR_d, phiR, numK*sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMemcpy(phiI_d, phiI, numK*sizeof(float), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMemset(Qi_d, 0, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cuda_ret = cudaMemset(Qr_d, 0, numX*sizeof(float));
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
}
