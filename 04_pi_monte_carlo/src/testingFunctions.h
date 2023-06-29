/** @file testingFunctions.h
 *
 *  @brief	File with test functions to check if \f$n\f$ points can be
 *		computed on the GPU memory.
 */

void limits(const int n){

  //Getting free and total memory available on device in bytes
  size_t freeMem, totalMem;
  CUDA_CALL( cudaMemGetInfo(&freeMem, &totalMem) );

  //Memory in GB, 1073741824 bytes (1024*1024*1024) in a GB
  const double freeMemGB = (double)freeMem / 1073741824;
  const double totalMemGB = (double)totalMem / 1073741824;

  //Memory needed to compute n points on GPU
  size_t nSizeInt = (n + n/BLOCK_SIZE + 1)*sizeof(int);

  const double sizeInGB = (double)nSizeInt / 1073741824;
  printf("Size of %d points: %.2lf GB\n",n,sizeInGB);
  if(sizeInGB > freeMemGB){
    printf("%.2lf GB exceeds the free memory on GPU %.2lf GB\n",sizeInGB,freeMemGB);

    //Computing the max value that can be used based on the available free memory
    int nMaxMem = freeMem/4.0;
    nMaxMem -= nMaxMem % BLOCK_SIZE;

    printf("%d is the max value for the available memory on GPU\n",nMaxMem);
    exit(1);
  }

  //Checking n to avoid exceeding max values of an int variable 
  const int maxValueInt = INT_MAX;
  if(n > maxValueInt){
    printf("%d exceeds the MAX int %d\n",n,maxValueInt);
    exit(1);
  }
}
/** \fn void limits(const int n)
 *
 * @brief	Host function to check limits when \f$n\f$ is computed on GPU.
 *
 * @details	The routine tests the free memory on device, checking if it is
 * 		enough to compute \f$n\f$ random pairs at the same time, if not,
 * 		it suggests a reasonable \f$n\f$-size and stops program execution.
 *
 * @param[in]   n	(Host) number of random pairs \f$n\f$
 */
