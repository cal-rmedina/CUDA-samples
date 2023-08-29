/** @file mandelbrot_ker.h 
 *
 * @brief  File with kernel to compute Mandelbrot set.
 */

template <typename var_type>
__global__ void mandelbrot_ker(
	const int width, const float2 ini, const float2 division,
	int *output){

  // Addressing pixels and index of the vector
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int i = y*width + x;

  // Corresponding complex number for the (x,y) pixel
  const var_type c_real = ini.x + (x+0.5)*division.x;
  const var_type c_imag = ini.y + (y+0.5)*division.y;

  // Initial complex number
  var_type real = 0.0, imag = 0.0,
           real2 = 0.0, imag2 = 0.0;

  int iter=0;
  while(iter<MAX_DWELL && real2+imag2 < 4.0f){
    real2 = real*real;
    imag2 = imag*imag;

    imag = 2.0f*real*imag + c_imag;
    real = real2 - imag2 + c_real;

    iter++;
  }
  output[i]=iter;
}
/** \fn void mandelbrot_ker(const int width,const float2 ini,const float2 division,int *output)
 *
 * @brief	2D kernel that processes individual pixels of the output image,
 * 		determining the number of iterations required for each pixel's
 * 		corresponding complex number to escape a predefined
 * 		threshold
 *
 * @details	The kernel calculates the pixel's coordinates (\f$x\f$ and
 * 		\f$y\f$) based on the  thread and block indices. It computes
 * 		the corresponding complex number for the pixel using the
 * 		provided ini and division parameters. If the iterator
 * 		exceeds the threshold \f$\vert z \vert < 2\f$ and the maximum
 * 		number of iterations \em MAX_DWELL, the loop concludes by
 * 		storing the iteration count in the output array
 *
 * @param[in]   width    width of the output image in pixels
 * @param[in]   ini      image origin in the complex plane for the Mandelbrot Set
 * @param[in]   division pixel dimensions in the complex plane \f$(x,y)\f$ of \f$ x + iy \f$
 * @param[out]  *output  (Device) array to store pixel count 
 */
