/** @file testingFunctions.h
 *
 *  @brief	File with test functions to check \f$n\f$ value and GPU errors.
 */

void limits(const int n){
  //Checking if n! is still inside the range to be computed
  if(n<0){
    printf("%d! is not defined for negative integers in mathematics\n",n);
    exit(1);
  }
  else if(n>20){
    unsigned long long max_value = ULLONG_MAX;
    printf("%d! exceeds the MAX unsigned long long %llu\n",n,max_value);
    printf("Use an integer number between [0,20]\n");
    exit(1);
  }
}
/** \fn void limits(const int n)
 *
 * @brief	Host function to check limits when \f$n!\f$ is computed.
 *
 * @details     The variable to store \f$n!\f$ is declared as unsigned long
 * 		long, which means that \f$n!\f$ value should not exceed the maximum
 * 		value stored in an unsigned long long variable, which in this
 * 		case is when \f$n>20\f$. If this value is exceeded, the program
 * 		stops.
 *
 * @param[in]   n	(Host) number to compute \f$n!\f$
 */
