/** @mainpage Addressing pixels with 2D arrays: GPU Documentation
 *
 * @section intro_sec Introduction
 * The Mandelbrot set is calculated by iteratively determining whether complex
 * numbers at each pixel on a 2D grid belong to the set. The code employs CUDA
 * parallelism, using 2D thread blocks and threads to process pixels
 * concurrently. The set's dimensions and complex plane boundaries are defined,
 * and the grid is divided accordingly, Mandelbrot kernel is launched. After
 * the computation, the results are transferred back to the CPU, and an output
 * file in the PPM format is generated to visualize the Mandelbrot set
 *
 * @section files_sec Files structure
 * The documentation is organized based on the following folder structure:
 * - @ref main.cu "Main"
 * - @ref src "Source"
 *
 * @subsection main_sub Main file
 * The \b main.cu file contains the core of the code. It is a good starting
 * point to see what the code does
 *
 * @subsection src_sub Source routines folder
 * The \b src folder contains all the files with the functions, kernels and
 * kernel calls used inside the program. Each time a new file with a routine is
 * created, its path should be added with the following format:
 * \verbatim
   #include "./src/file_name"
   \endverbatim
 *
 * @section flags_sec Flags
 * The program displays image details on the screen when the `PRINT_SET_DETAILS`
 * flag is enabled, either directly in the Makefile or by entering the option
 * during compilation in the terminal
 *
 */

/** @file main.cu 
 * 
 * @brief Main code including the libraries, dependences, source files and
 * main function
 */

#include <stdio.h>
#include <stdlib.h>

#define MAX_DWELL 512
#define thrX 16
#define thrY 16

/**
 * @def MAX_DWELL
 * @brief Maximum dwell value used in the computation.
 *
 * @def thrX
 * @brief Number of threads in the X dimension of a thread block.
 *
 * @def thrY
 * @brief Number of threads in the Y dimension of a thread block.
 */

#include "./src/cudaMacros.h"
#include "./src/testingFunctions.h"
#include "./src/mandelbrot_ker.h"

int main(){

  // Number of pixels
  const unsigned int width  = 1 << 10;
  const unsigned int height = 1 << 10;

  // Test functions for the limits 
  limits(width,height);

  // Limits complex plane
  const float2 ini = make_float2(-2.25f,-1.5f),
	       fin = make_float2( 1.25f, 1.5f);

  // Increments (dx, dy)
  const float2 div = make_float2((fin.x-ini.x)/width,(fin.y-ini.y)/height);

  #ifdef PRINT_SET_DETAILS
    // Screen output with image specifications 
    printf("Mandelbrot set: mb_rgn.ppm)\n");
    printf("Size: %dx%dpx\n", width, height);
    printf("Complex plane (x,y) with c = x+iy:\n");
    printf("x in [%.2f,%.2f], dx = %f\n", ini.x, fin.x, div.x);
    printf("y in [%.2f,%.2f], dy = %f\n", ini.y, fin.y, div.y);
  #endif

  // Size of the output vector
  size_t size_vec = width*height*sizeof(int);

  int* d_dwells;
  CUDA_CALL( cudaMalloc(&d_dwells,size_vec) );

  // 2D-block & 2D-thread size
  dim3 threadsPerBlock(thrX,thrY);
  dim3 numBlocks( width/threadsPerBlock.x, height/threadsPerBlock.y); 

  // Kernel call
  mandelbrot_ker<float><<<numBlocks,threadsPerBlock>>>(width,ini,div,d_dwells);
  CUDA_CALL( cudaPeekAtLastError() );

  // Host vector and copy from device to host
  int *h_dwells = (int *)malloc(size_vec);
  CUDA_CALL( cudaMemcpy(h_dwells,d_dwells,size_vec,cudaMemcpyDeviceToHost) );

  // Generating output file
  FILE *out_rgb_file = fopen("mb_rgb.ppm","w");
  // Image header for ppm format
  fprintf(out_rgb_file, "P3\n%d %d\n%d\n", width, height, 256);
  for(unsigned int y=0; y<height; y++){
    for(unsigned int x=0; x<width; x++){
      unsigned int i = h_dwells[y*width+x];
      int r = i % 128 *  2; 
      int g = i %  64 *  4;
      int b = i %  32 *  8;
      fprintf(out_rgb_file,"%d %d %d\n",r,g,b);
    }
  }
  fclose(out_rgb_file);
  free(h_dwells);

  CUDA_CALL( cudaFree(d_dwells) );

  return 0;
}
/** \fn int main()
 *
 * @brief       Main function calling the other functions
 *
 * @details     Allocating/freeing memory for the vector \em vec[\f$
 * 		width\times height\f$] to store the pixel value computed on
 * 		Device (GPU), the vector stores data indexig each pixel with
 * 		the formula \f$ y\times width + x\f$
 *
 * @return      An integer 0 upon exit success
 */
