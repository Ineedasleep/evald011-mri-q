/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>

#include "parboil.h"

#include "file.h"
#include "computeQ.cc"
#include "computeQ.cu"
#include "support.h"

int
main (int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */
  struct kValues* kVals;

  float *x_d, *y_d, *z_d; /* Device X coordinates (3D vectors) */
  float *phiR_d, *phiI_d; /* Device Phi values (complex) */
  float *phiMag_d; /* Device Phi Magnitude */
  float *Qr_d, *Qi_d; /* Device Q signal (complex) */
  struct kValues* kVals_d; /* Device K trajectory (3D vector + magnitude) */

  cudaError_t cuda_ret;

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {

      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
  
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);
  /* Create GPU data structures */  
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  // Note: the below function also performs memcpys of x, y, z, phiR, and phiI
  createDataStructsGPU(numK, numX, x, y, z, phiR, phiI, x_d, y_d, z_d, phiR_d, phiI_d, phiMag_d, Qr_d, Qi_d, kVals_d);
  
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  ComputePhiMagCPU(numK, phiR, phiI, phiMag);
  
  /********** Testing process shows this is slower on GPU than CPU **********/
  
  // pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  // ComputePhiMagGPU(numK, phiR_d, phiI_d, phiMag_d);
  // pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  // cuda_ret = cudaMemcpy(phiMag, phiMag_d, numK*sizeof(float), cudaMemcpyDeviceToHost);
  // if (cuda_ret != cudaSuccess) {
  //     printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
  //     exit(EXIT_FAILURE);
  // }
    

  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  // Copying kVals to kVals_d
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  cuda_ret = cudaMemcpy(kVals_d, kVals, numK*sizeof(struct kValues), cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }

  //ComputeQCPU(numK, numX, kVals, x, y, z, Qr, Qi);

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  ComputeQGPU(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  cuda_ret = cudaMemcpy(Qr, Qr_d, numX*sizeof(float), cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }
  
  cuda_ret = cudaMemcpy(Qi, Qi_d, numX*sizeof(float), cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(cuda_ret), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
  }
  

  if (params->outFile)
    {
      /* Write Q to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);
  free (Qr);
  free (Qi);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(phiR_d);
  cudaFree(phiI_d);
  cudaFree(phiMag_d);
  cudaFree(Qr_d);
  cudaFree(Qi_d);
  cudaFree(kVals_d);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}
