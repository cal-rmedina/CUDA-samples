/** @file cudaMacros.h 
 * 
 * @brief File containing macro that is used handle potential errors 
 */

/**
 * @brief The purpose of this macro is to simplify error handling for CUDA
 * function calls. 
 *
 * @details Any CUDA function call can be wrapped with it, and it will
 * automatically check if the call succeeded. If it didn't, it will print an
 * error message indicating the CUDA error string, error code, line number, and
 * file name where the error occurred.
 * 
 * @param call  Result of the CUDA function.
 */

#define CUDA_CALL(call)		\
{				\
  cudaError_t result = call;	\
  if(cudaSuccess != result)	\
    fprintf(stderr,"CUDA error %s(%d) at line %d of %s\n",	\
      cudaGetErrorString(result), result, __LINE__, __FILE__);	\
}
