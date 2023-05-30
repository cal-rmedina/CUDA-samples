/** @file recursiveFactorialGPU.h
 *
 *  @brief	File with the device functions, kernels & kernel calls
 * 		computing \f$n!\f$ recursively using a device function.
 */

__device__ unsigned long long mult_n_rec(const int n){
  if(n<=1)  return 1;
  return n*mult_n_rec(n-1);
}
/** \fn unsigned long long mult_n_rec(const int n)
 *
 * @brief       Device funtion to compute recursively \f$n!\f$ using recursive device function. 
 *
 * @details     Function must be computed on device called from a kernel.
 *
 * @param[in]   n	(Device) number to compute \f$n!\f$
 *
 * @return	The product \f$n(n-1)\f$ or 1 when \f$n\leq1\f$
 */

__global__ void recursiveFactorialKernel(const int n,unsigned long long* d_vec){
  unsigned long long recursiveFact = mult_n_rec(n);
  d_vec[n-1] = recursiveFact;
}
/** \fn void recursiveFactorialKernel(const int n,unsigned long long* d_vec)
 *
 * @brief       Kernel calling device function to compute \f$n!\f$. 
 *
 * @details     Compute factorial with a recursive device function for each thread.
 *
 * @param[in]	n	(Device) number to compute \f$n!\f$
 * @param[out]	d_vec	(Device) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 */

void recursiveFactorialGPU(const int n,unsigned long long* h_vec,
	unsigned long long* d_vec){

  //Create n stream to overlap kernels while they are executed
  cudaStream_t streams[n];
  for(int i=0; i<n; i++)  cudaStreamCreate(&streams[i]);

  cudaEvent_t start,stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  //\f$n\f$ kernel calls, each with its respective stream
  for(int i=0; i<n; i++)  recursiveFactorialKernel<<<1,1,0,streams[i]>>>(i+1,d_vec);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds,start,stop);
  printf("Time (ms): Recursive device function GPU = %f\n",milliseconds);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for(int i=0; i<n; i++)  cudaStreamDestroy(streams[i]);

  #ifdef PRINT_FACTORIAL
    size_t n_SizeULL = n*sizeof(unsigned long long);
    cudaMemcpy(h_vec,d_vec,n_SizeULL,cudaMemcpyDeviceToHost);
    for(int i=1;i<=n;i++)  printf("%d!=%llu\n",i,h_vec[i-1]);
  #endif
}
 /** \fn void recursiveFactorialGPU(const int n,unsigned long long* h_vec,
 *		unsigned long long* d_vec)
 *
 * @brief       Host function calling kernel to compute \f$n!\f$ using recursive device funtion.
 *
 * @details     Kernel call is executed using one block with one thread;
 * 		\verbatim <<<1,1,0,streams[i]>>> \endverbatim
 *		being the first two entries the number of blocks in a grid
 *		(gridDim.x) and the number of threads in a block
 *		(BlockDimIdx.x) respectively, the 3rd and 4th entries are
 *		shared memory (not used in this case) and the respective stream
 *		assigned to the kernel execution.
 *
 * @param[in]	n	(Host) number to compute \f$n!\f$
 * @param[out]	h_vec	(Host) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 * @param[out]	d_vec	(Device) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 */
