/** @mainpage Monte-Carlo for estimating pi value; GPU Documentation
 *
 * @section intro_sec Introduction
 * Estimating \f$\pi\f$-value computing \f$ n \f$ random points, each random
 * point \f$ (x,y) \f$ with \f$x,y\in[0,1]\f$ is tested to see if it falls
 * inside a circumference \f$r=1\f$. 
 *
 * @section files_sec Files structure
 * The documentation is organized based on the following directory structure:
 * - Root folder containing the main file @ref main.cu
 * - @ref src folder with code fuctions
 * - @ref manual directory
 *
 * @subsection main_sub Main file
 * The \b main.cu file contains the core of the code and function calls. It is
 * a good starting point to see what the code does.
 *
 * @subsection src_sub Source routines folder
 * The src folder contains all the files with the functions, kernels and
 * kernel calls used inside the program. Each time a new file with a routine is
 * created, its path should be added with the following format:
 * \verbatim
   #include "./src/file_name"
   \endverbatim

 * @subsection manual_sub General manual
 * Detailed manual written in \f$\mbox{\LaTeX}\f$ has been developed and can be
 * found in the root directory, check its respective chapter for detailed info
 * about the equations and algorithms used to solve the problem.
 */

/** @file main.cu 
 * 
 * @brief Main code including the libraries, dependences, source files and
 * main function.
 */

//C libraries used
#include <stdio.h>
#include <limits.h>

//CUDA libraries
#include <cuda.h>
#include <curand_kernel.h>

//1D Block size for the kernel
#define  BLOCK_SIZE 256

//Routines created
#include "./src/cudaMacros.h"
#include "./src/testingFunctions.h"
#include "./src/piMonteCarloCPU.h"
#include "./src/piMonteCarloGPU.h"
#include "./src/reduction6GPU.h"

//Main function
int main(){

  //Bitwise shift operator, (Appendix manual for more info)
  const int nBlocks = 1<<21;

  //Number of samples to compute pi, multiple of BLOCK_SIZE for more efficient reductions
  const int n = nBlocks*BLOCK_SIZE;

  //Serial host function computing pi on CPU
//  piMonteCarloCPU(n);

  //Test available memory on GPU
  limits(n);

  //Allocating space for vector with results (Device)
  size_t nSizeInt = n*sizeof(int);
  int *d_result;	//Device result vector
  CUDA_CALL( cudaMalloc(&d_result,nSizeInt) );

  //Computing n random number (x,y) and checking x*x + y*y <= 1
  innerCircleGPU(n,d_result);

  //Allocating space for reduction vector (Device)
  size_t nBlockSizeInt = nBlocks*sizeof(int);
  int *d_reduction;
  CUDA_CALL( cudaMalloc(&d_reduction,nBlockSizeInt) );

  reduce6GPU(n,nBlocks,d_result,d_reduction);

  //Freeing space for vector with results (Device)
  CUDA_CALL( cudaFree(d_result) );

  return 0;
}
/** \fn int main()
 *
 * @brief	Main function calling the other functions.
 *
 * @details	Allocating/freeing memory for the vector to store
 *		the results of \f$n!\f$ on host and device \em h_result and \em
 *		d_result respectively, and the different functions to compute
 *		\f$n!\f$.
 *
 * @return	An integer 0 upon exit success
 */
