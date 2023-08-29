/** @file testingFunctions.h
 *
 *  @brief	File with test functions to check if \f$n\f$ points
 *		(width*height) can be computed on the GPU memory.
 */

void limits(const unsigned int width, const unsigned int height){

  // Getting free and total memory available on device in bytes
  size_t freeMem, totalMem;
  CUDA_CALL( cudaMemGetInfo(&freeMem, &totalMem) );

  // Memory in GB, 1073741824 bytes (1024*1024*1024) in a GB
  const double freeMemGB = (double)freeMem / 1073741824;
  const double totalMemGB = (double)totalMem / 1073741824;

  // Memory needed to compute width*height points on GPU
  size_t nSizeInt = width*height*sizeof(int);

  // Memory (GB and MB))
  double sizeInGB = (double)nSizeInt / 1073741824;
  double sizeInMB = (double)nSizeInt / 1048576;

  // Printing sized of the image
  printf("Image size: %d x %d pixels (%d points)\n",width,height,width*height);
  if(sizeInGB < 0.01)  printf("Size of %d points: %.2lf MB\n",width*height,sizeInMB);
  if(sizeInGB > 0.01)  printf("Size of %d points: %.2lf GB\n",width*height,sizeInGB);

  // Exit if the value exceeds the available memory
  if(sizeInGB > freeMemGB){
    printf("%.2lf GB exceeds the free memory on GPU %.2lf GB\n",sizeInGB,freeMemGB);

    // Computing the max value that can be used based on the available free memory
    int nMaxMem = freeMem/4.0;

    printf("%d is the max numb. of pixels for the available memory on GPU\n",nMaxMem);
    exit(1);
  }

  // Checking n to avoid exceeding max values of an int variable 
  const int maxValueInt = INT_MAX;
  if(width*height > maxValueInt){
    printf("%d pixels exceed the MAX int %d\n",width*height,maxValueInt);
    exit(1);
  }
}
/** \fn void limits(const unsigned int width, const unsigned int height)
 *
 * @brief	Host function to check limits when \f$n\f$ is computed on GPU.
 *
 * @details	The routine tests the free memory on device, checking if it is
 * 		enough to compute width \f$\times\f$ height random pairs at the
 * 		same time, if not, it suggests a reasonable \f$n\f$-size
 * 		(width*height) and stops program execution.
 *
 * @param[in]   width	(Host) width of image (px)
 * @param[in]   weight	(Host) height of image (px)
 */
