/** @mainpage Factorial GPU Documentation
 *
 * @section intro_sec Introduction
 * Computing the factorial of an integer number \f$ n! \f$ computed with
 * different approximations to test efficiency of the implemented routines.
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
#include <limits.h>

//Routines created
#include "./src/testingFunctions.h"
#include "./src/factorialGPU.h"
#include "./src/recursiveFactorialGPU.h"
#include "./src/serialFactorialCPU.h"

//Main function
int main(){

  const int n = 20;	//Factorial computed for 1!, 2!,..., 20!
  
  //Test function
  limits(n);

  //Allocating space for vector with results (Host and device)
  size_t n_SizeULL = n*sizeof(unsigned long long);
  unsigned long long *h_result, *d_result;	//Host and device result vector
  h_result =(unsigned long long *)malloc(n_SizeULL);
  cudaMalloc(&d_result,n_SizeULL);

  //Kernel call computing factorial on GPU
  factorialGPU(n,h_result,d_result);

  //Kernel call computing factorial with a recursive device function on GPU
  recursiveFactorialGPU(n,h_result,d_result);

  //Linear host function computing factorial on CPU
  factorialCPU(n,h_result);

  //Freeing space for vector with results (Host and device)
  free(h_result);
  cudaFree(d_result);

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
