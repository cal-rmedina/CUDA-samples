/** @file squareKernel.h 
 * 
 * @brief Main code including the libraries, dependences, source files and
 * main function.
 */

__global__ void square_ker(int *d_vec, const unsigned int n){

  unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
  for(int i = tid; i<n; i += blockDim.x*gridDim.x)  d_vec[i]=(i+1)*(i+1);
}
/** \fn void square_ker(int *d_vec, const unsigned int n)
 *
 * @brief       Kernel to compute \f$ (i+1)^2 \f$
 *
 * @details     Each thread computes \em vec[i]=\f$(i+1)^2 \f$
 *
 * @param[out]	*d_vec	(Device) vector to store \f$(i+1)^2\f$
 * @param[in]   n	number of elements of vector \f$n\f$
 */

