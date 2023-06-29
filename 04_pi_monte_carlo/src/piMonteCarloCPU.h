/** @file piMonteCarloCPU.h 
 *
 * @brief	File with host function to compute \f$\pi\f$ directly on
 *		the CPU (serial) just for reference.
 */

/**
 * @brief 	Macro for generating a random double value
 *		\f$\in\left[0,1\right]\f$.
 *
 *		This macro uses the `rand()` function from the C standard
 *		library to generate a random integer value, divides it by the
 *		maximum possible value `RAND_MAX`, and converts the result to a
 *		double value \f$\in\left[0,1\right]\f$.
 */
#define RND1 ((double)((double)rand()/(double)RAND_MAX))

void piMonteCarloCPU(const int n){

  //Random numbers with time seed to make a different sequence every run
  time_t seed_t;
  srand((unsigned) time(&seed_t));	//Init. random number generator

  //Counter for the numbers inside the circle
  unsigned long long count = 0;

  for(int i=0; i<n; i++){
    float x_ran = RND1, y_ran = RND1;
    float radius = x_ran*x_ran + y_ran*y_ran; 
    if(radius <= 1.0)  count++;
  }
  const double pi = 4.0*count/n;

  #ifdef PRINT_PI
    printf("pi = %.16lf computed on the CPU for %d samples\n",pi,n);
  #endif
}
/** \fn void piMonteCarloCPU(const int n)
 *
 * @brief	Host function to compute \f$\pi\f$ on the CPU.
 *
 * @details     Used to compute \f$\pi\f$ with computing \f$ n \f$ random pairs
 * 		\f$(x,y)\in\left[0,1\right]\f$.
 *
 * @param[in]   n	(Host) number of random pairs \f$n\f$
 */
