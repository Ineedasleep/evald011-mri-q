#include "computeQ.cc"

#define MU_THREADS_PER_BLOCK 256
#define Q_THREADS_PER_BLOCK 256

/*
Original:

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
    indexK = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float real = phiR[indexK];
        float image = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
}

void ComputePhiMagGPU(int numK, float* phiR, float* phiI, float* __restrict__ phiMag) {
    dim3 dim_grid((numK-1)/MU_THREADS_PER_BLOCK + 1, 1, 1);
    dim3 dim_block(MU_THREADS_PER_BLOCK, 1, 1);

    dev_ComputePhiMagGPU<<<dim_grid, dim_block>>>(numK, phiR, phiI, phiMag);
}

/*
//inline
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

__global__
void dev_ComputeQGPU(const int numK, const int numX,
                    struct kValues *kVals,
                    const float* x, const float* y, const float* z,
                    float* Qr, float* Qi) {
    // Local vars
    float loc_x, loc_y, loc_z;
    float Qracc = 0.0f;
    float Qiacc = 0.0f;

    // Find index of voxel assigned to this thread
    int indexX = blockIdx.x * Q_THREADS_PER_BLOCK + threadIdx.x;
    
    __shared__ struct kValues kVals_tile[Q_THREADS_PER_BLOCK];
    
    for(int i = 0; i < (numK-1)/Q_THREADS_PER_BLOCK + 1; i++) { // Loop for each tile
        //Collaborative loading
        if(indexX < numX) {
            loc_x = x[indexX];
            loc_y = y[indexX];
            loc_z = z[indexX];
            kVals_tile[threadIdx.x] = kVals[i*Q_THREADS_PER_BLOCK+threadIdx.x];
        } else {
            loc_x = 0;
            loc_y = 0;
            loc_z = 0;
            kVals_tile[threadIdx.x] = 0;
        }
        __syncthreads(); // sync to ensure tile properly loaded
    
        // Accumulation
        if(indexX < numX) { // Checking data bounds
            for (indexK = 0; indexK < Q_THREADS_PER_BLOCK; indexK++) {
                expArg = PIx2 * (kVals_tile[indexK].Kx * loc_x +
                                kVals_tile[indexK].Ky * loc_y +
                                kVals_tile[indexK].Kz * loc_z);

                cosArg = cosf(expArg);
                sinArg = sinf(expArg);

                float phi = kVals_tile[indexK].PhiMag;
                Qracc += phi * cosArg;
                Qiacc += phi * sinArg;    
            }
        }
    }

    if(indexX < numX) {
        Qr[indexX] = Qracc;
        Qi[indexX] = Qiacc;
    }
}

void
ComputeQGPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *__restrict__ Qr, float *__restrict__ Qi) {
    dim3 dim_grid((numK-1)/Q_THREADS_PER_BLOCK + 1, 1, 1);
    dim3 dim_block(Q_THREADS_PER_BLOCK, 1, 1);

    dev_ComputeQGPU(numK, numX, kVals, x, y, z, Qr, Qi);
}

/*
Original:
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

void createDataStructsGPU(int numK, int numX, float** phiMag, float** Qr, float** Qi)) {    
    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc((void**) &(*phiMag), numK*sizeof(float));
    if (cuda_ret != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void**) &(*Qr), numX*sizeof(float));
    if (cuda_ret != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemset((void *)*Qr, 0, numX * sizeof(float));
    if (cuda_ret != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMalloc((void**) &(*Qi), numX*sizeof(float));
    if (cuda_ret != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cuda_ret = cudaMemset((void *)*Qi, 0, numX * sizeof(float));
    if (cuda_ret != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}
