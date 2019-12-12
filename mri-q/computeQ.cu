

#define MU_THREADS_PER_BLOCK 256

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
void ComputePhiMagGPU(int numK, float* phiR, float* phiI, float* __restrict__ phiMag) {
    dim3 dim_grid((numK-1)/MU_THREADS_PER_BLOCK + 1, 1, 1);
    dim3 dim_block(MU_THREADS_PER_BLOCK, 1, 1);

    dev_ComputePhiMagGPU<<<dim_grid, dim_block>>>(numK, phiR, phiI, phiMag);
}

__global__
void dev_ComputePhiMagGPU(const int numK, const float* phiR, const float* phiI, float* phiMag) {
    indexK = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float real = phiR[indexK];
        float image = phiI[indexK];
        phiMag[indexK] = real*real + imag*imag;
    }
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
    float *A_d, *B_d, *C_d;
    
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