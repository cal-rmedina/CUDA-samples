/** @file factorialGPU.h
 *
 * @brief	File with the kernel & kernel call computing \f$n!\f$.
 */

__global__ void factorialKernel(const int n,unsigned long long* d_vec){

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){
    unsigned long long fact = 1;
    for(int j=2; j<i+2; j++)  fact*=j;
    d_vec[i] = fact;
  }
}
/** \fn void factorialKernel(const int n,unsigned long long* d_vec)
 *
 * @brief       Kernel to compute \f$n!\f$. 
 *
 * @details     Used to compute factorial with a for loop for each thread.
 *
 * @param[in]   n	(Device) number to compute \f$n!\f$
 * @param[out]	d_vec	(Device) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 */

void factorialGPU(const int n,unsigned long long* h_vec,
	unsigned long long* d_vec){

  cudaEvent_t start,stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  //Kernel call
  factorialKernel<<<1,n>>>(n,d_vec);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds,start,stop);
  printf("Time (ms): Kernel call loop per thread GPU = %f\n",milliseconds);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  #ifdef PRINT_FACTORIAL
    size_t n_SizeULL = n*sizeof(unsigned long long);
    cudaMemcpy(h_vec,d_vec,n_SizeULL,cudaMemcpyDeviceToHost);
    for(int i=1;i<=n;i++)  printf("%d!=%llu\n",i,h_vec[i-1]);
  #endif
}
 /** \fn void factorialGPU(const int n,unsigned long long* h_vec,
 *		unsigned long long* d_vec)
 *
 * @brief       Host function calling kernel to compute \f$n!\f$.
 *
 * @details     Kernel call is executed using one block with \f$n\f$ threads;
 * 		\verbatim <<<1,n>>> \endverbatim 
 *		being the first two entries the number of blocks in a
 * 		grid (gridDim.x) and the number of threads in a block
 * 		(BlockDimIdx.x) respectively.
 *
 * @param[in]	n	(Host) number to compute \f$n!\f$
 * @param[out]	h_vec	(Host) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 * @param[out]	d_vec	(Device) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 */
