/** @mainpage Streams GPU Documentation
 *
 * @section intro_sec Introduction
 * This section provides an overview of the code's purpose, which involves
 * calculating the sum of the first \f$ n \f$ square natural numbers
 * (\f$1+2+\dots+n\f$). The primary focus of this program is not on efficiency
 * but on illustrating the **utilization of concurrent streams** and the NVIDIA
 * Nsight profiler. In this context, each kernel call is designed to involve
 * only a single thread.
 *
 * @section files_sec Files structure
 * The documentation is organized based on the following folder structure:
 * - @ref main.cu "Main"
 * - @ref src "Source"
 *
 * @subsection main_sub Main file
 * The \b main.cu file contains the core of the code. It is a good starting
 * point to see what the code does.
 *
 * @subsection src_sub Source routines folder
 * The \b src folder contains all the files with the functions, kernels and
 * kernel calls used inside the program. Each time a new file with a routine is
 * created, its path should be added with the following format:
 * \verbatim
   #include "./src/file_name"
   \endverbatim
 */

/** @file main.cu 
 * 
 * @brief Main code including the libraries, dependences, source files and
 * main function.
 */

//C libraries used
#include <stdio.h>

//Routines created
#include "./src/cudaMacros.h"
#include "./src/squareKernel.h"

//Main function
int main(){

  const unsigned int N =1<<8;  // Number of elements

  const unsigned int N_streams = 1<<2;  // Number of streams
  cudaStream_t streams[N_streams];

  // Vector declaration, size & allocation
  size_t n_size_int = N*sizeof(int);

  int *d_vec[N_streams];

  for(unsigned int i=0; i<N_streams; i++){

    CUDA_CALL( cudaStreamCreate(&streams[i]) );

    CUDA_CALL( cudaMalloc(&d_vec[i], n_size_int) );

    // Launch one kernel per stream
    square_ker<<<1, N, 0, streams[i]>>>(d_vec[i],N);
    CUDA_CALL( cudaPeekAtLastError() );
  }

  // Destroying the streams

  for(unsigned int i=0; i<N_streams; i++)  CUDA_CALL( cudaStreamDestroy(streams[i]) );

  #ifdef PRINT_1ST_STREAM
    // Check results on device and compare with results on host
    int h_vec[N];
    CUDA_CALL( cudaMemcpy(h_vec,d_vec[0],n_size_int,cudaMemcpyDeviceToHost) );

    unsigned int sumGPU = 0, sumCPU = 0;
    for(unsigned int i=0; i<N; i++){
      sumGPU += h_vec[i];
      sumCPU += pow(i+1,2);
      if(h_vec[i] != (i+1)*(i+1))  printf("%d %d %d\n", i+1, h_vec[i], (i+1)*(i+1));
    }
  
    // Print sum on host and device
    printf("Sum of first %d square naturals: CPU=%d, GPU=%d\n",N,sumCPU,sumGPU);
  #endif

  return 0;
}
/** \fn int main()
 *
 * @brief	Main function calling the other functions.
 *
 * @details	Allocating/freeing memory for the vector \em vec[n] to store
 * 		the results of \em vec[i] \f$=\left(i+1\right)^2\f$ with
 * 		\f$ i \in \left[0,N \right) \f$ on host and device \em h_vec
 * 		and \em d_vec respectively.
 *
 * @return	An integer 0 upon exit success
 */
