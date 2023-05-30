/** @file serialFactorialCPU.h 
 *
 * @brief	File with host function to compute serial \f$n!\f$ directly on
 *		the CPU.
 */

void factorialCPU(const int n,unsigned long long* h_vec){
  unsigned long long fact = 1;
  h_vec[0] = fact;
  for(int i=2; i<=n; i++){
    fact *= i;
    h_vec[i-1] = fact;
  }

  #ifdef PRINT_FACTORIAL
    for(int i=0;i<n;i++)  printf("%d!=%llu\n",i+1,h_vec[i]);
  #endif
}
/** \fn void factorialCPU(const int n,unsigned long long* h_vec)
 *
 * @brief	Host function to compute \f$ n!\f$ on the CPU.
 *
 * @details     Used to compute factorial with a direct for loop.
 *
 * @param[in]   n	(Host) number to compute \f$n!\f$
 * @param[out]	h_vec	(Host) vector to store \f$n!\f$, i.e. \f$v_{res}[i]=(i+1)!\f$
 */
