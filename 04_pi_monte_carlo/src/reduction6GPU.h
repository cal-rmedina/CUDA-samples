/** @file reduction6GPU.h 
 * 
 * @brief Block reduction to add up all the elements of a vector.
 */

template <unsigned int blockSize>
__global__ void reduce6(const int *g_idata, int *g_odata, size_t n){
  extern volatile __shared__ int sMemVec[];
  size_t  tid		= threadIdx.x,
	  gridSize	= blockSize * gridDim.x,
          i		= blockIdx.x * blockSize + tid;
  sMemVec[tid] = 0;

  while(i < n){
    sMemVec[tid] += g_idata[i];
    i += gridSize;
  }
  __syncthreads();

  if(blockSize >= 512){
    if(tid < 256)  sMemVec[tid] += sMemVec[tid + 256];
     __syncthreads();
   }
  if(blockSize >= 256){
    if(tid < 128)  sMemVec[tid] += sMemVec[tid + 128];
     __syncthreads();
  }
  if(blockSize >= 128){
    if(tid < 64)  sMemVec[tid] += sMemVec[tid + 64];
    __syncthreads();
  }

  if(tid <32){
    if(blockSize >= 64)  sMemVec[tid] += sMemVec[tid + 32];
    if(blockSize >= 32)  sMemVec[tid] += sMemVec[tid + 16];
    if(blockSize >= 16)  sMemVec[tid] += sMemVec[tid + 8];
    if(blockSize >= 8)   sMemVec[tid] += sMemVec[tid + 4];
    if(blockSize >= 4)   sMemVec[tid] += sMemVec[tid + 2];
    if(blockSize >= 2)   sMemVec[tid] += sMemVec[tid + 1];
  }
  if (tid == 0) g_odata[blockIdx.x] = sMemVec[0];
}
/** \fn void reduce6(const int *g_idata, int *g_odata, size_t n)
 *
 * @brief       Kernel to reduce (sum over all the elements) a \f$n\f$-size vector. 
 *
 * @details     Perform a \f$n\f$ reduction adding up all the elements of the vector,
 * 		each block performs a reduction writing the sum (of the threads
 * 		inside that block) in the output vector \b g_odata, the output
 * 		vector should have the size of the number of blocks used to
 * 		perform the reduction.
 *
 * @param[in]   g_idata	(Device) input vector to be block reduced
 * @param[out]	g_odata (Device) output vector with the respective block sums in each entry
 * @param[in]	n (Device) \b g_idata size
 */

void reduce6GPU(const int n, const int nBlocks, const int* d_vec, int* d_red){

  int *d_sum;
  int sum;
  //Memory allocation (Device) for the final sum value (sum of d_vec elements)
  CUDA_CALL( cudaMalloc(&d_sum,sizeof(int)) );

  //Shared memory used in each Block reduction
  size_t sharMemSize = BLOCK_SIZE*sizeof(int);

  //First kernel call to reduce the vector d_vec and store the elements in d_red
  reduce6<BLOCK_SIZE><<<nBlocks,BLOCK_SIZE,sharMemSize>>>(d_vec,d_red,n);
  CUDA_CALL( cudaPeekAtLastError() );

  //Second kernel call to reduce the vector d_red and store its value in d_sum
  reduce6<BLOCK_SIZE><<<1,BLOCK_SIZE,sharMemSize>>>(d_red,d_sum,nBlocks);
  CUDA_CALL( cudaPeekAtLastError() );

  CUDA_CALL( cudaDeviceSynchronize() );

  //Copy d_sum value to the host once the sums are completed
  CUDA_CALL( cudaMemcpy(&sum,d_sum,sizeof(int),cudaMemcpyDeviceToHost) );
  CUDA_CALL( cudaFree(d_sum));

  const double pi = 4.0*sum/n;
  
  #ifdef PRINT_PI
    printf("pi = %.16lf computed on the GPU for %d samples\n",pi,n);
  #endif
}
/** \fn void reduce6GPU(const int n, const int nBlocks, const int* d_vec, int* d_red)
 *
 * @brief       Host function with kernel calls to reduce (sum over all the
 * 		elements) a \f$n\f$-size vector. 
 *
 * @details     Calls two times the kernel to reduce a \f$n\f$-size vector,
 * 		first kernel call reduces the \f$n\f$-vector to a 
 * 		nBlocks-vector, second call reduces the nBlocks-vector to a
 * 		escalar.
 *
 * @param[in]	n 	(Device) \b d_vec size
 * @param[in]	nBlocks (Device) \b d_red size
 * @param[in]   d_vec (Device) input vector to be block reduced
 * @param[out]	d_red (Device) output vector with the respective block sums in each entry
 */
