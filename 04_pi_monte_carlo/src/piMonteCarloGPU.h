/** @file piMonteCarloGPU.h 
 *
 * @brief	File with device function kernel and kernel call to compute
 * 		\f$n\f$ points on the GPU.
 */

__global__ void innerCircleKernel(const int n, const int seed,int* d_vec){

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i<n){

    //Initialize the cuRAND random number generator
    curandState state;
    curand_init(seed,i,0,&state);

    d_vec[i]=0;		//Setting number to 0

    //Generate pseudo-random uniform random numbers (0,1]
    float x_ran = curand_uniform(&state);
    float y_ran = curand_uniform(&state);
 
    float radius = x_ran*x_ran + y_ran*y_ran; 
    if(radius < 1.0)  d_vec[i]=1;
  }
}
/** \fn void innerCircleKernel(const int n, const int seed,int* d_vec)
 *
 * @brief       Kernel to compute n points with a random seed. 
 *
 * @details     Initialize \f$n\f$ curandState used to generate two random numbers
 * 		on each  thread, create \f$n\f$ random points (x,y) inside the
 * 		positive quadrant; \f$x,y \in [0,1]\f$. Random pairs are tested
 * 		and added to the vector if they fall inside the circle.
 *
 * @param[in]   n	(Device) number of points to compute \f$\pi\f$
 * @param[in]	seed	(Device) same random seed for each curandState 
 * @param[out]	d_vec	(Device) vector storing numbers inside circle (0 or 1 outside/inside). 
 */

void innerCircleGPU(const int n, int* d_vec){

  dim3 blockDim (BLOCK_SIZE);
  //The number of Blocks is chosen as follows cause n is a multiple of BLOCK_SIZE
  dim3 gridDim  (n/blockDim.x);

  int seed = rand();

  //Kernel call
  innerCircleKernel<<<gridDim,blockDim>>>(n,seed,d_vec);
  CUDA_CALL( cudaPeekAtLastError() );
}
/** \fn void innerCircleGPU(const int n, int* d_vec)
 *
 * @brief	Host function calling kernel to generate \f$n\f$ random numbers.
 *
 * @details     Dimension of the blocks and threads is given by the number of points \f$n\f$,
 *		in this case \f$n\f$ is defined as a multiple of BLOCK_SIZE,
 *		therefore the number of threads and blocks is BLOCK_SIZE and
 *		n/BLOCK_SIZE respectively. 
 *
 * @param[in]   n	(Host) number of random pairs \f$n\f$
 * @param[out]	d_vec	(Device) vector storing numbers inside circle (0 or 1 outside/inside). 
 */
