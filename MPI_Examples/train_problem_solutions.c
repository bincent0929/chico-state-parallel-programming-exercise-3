/* File:     mpi_trap3.c
 * Purpose:  Use MPI to implement a parallel version of the trapezoidal 
 *           rule.  This version uses collective communications to 
 *           distribute the input data and compute the global sum.
 *
 * Input:    The endpoints of the interval of integration and the number
 *           of trapezoids
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Compile:  Run the make
 * Run:      mpiexec -n <number of processes> ./mpi_trap3_left_riemann_ten_ms
 *
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 * Note:  f(x) is all hardwired.
 *
 * IPP:   Section 3.4.2 (pp. 104 and ff.)
 */

// This is required for M_PI to be loaded in Linux
#define _GNU_SOURCE
/**
 * For Part 2:
 *    Set to PART2 + VELOCITY || DISPLACEMENT
 * For Part 3:
 *    Set to PART3 + VELOCITY || DISPLACEMENT
 * (HALF_TIME is used to check whether the derivation of velocity
 *  from acceleration actually produces the correct value by getting the peak value).
 */
#define PART4
//#define HALF_TIME
#define DISPLACEMENT

#include <stdio.h>
#include <math.h>

/* We'll be using MPI routines, definitions, etc. */
#include <mpi.h>

/* Get the input values */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p,
      int* n_p);

/* Calculate local integral  */
double Trap(double left_endpt, double right_endpt, int trap_count, 
   double base_len);    
float LeftRiemann(float left_endpt, float right_endpt, int rect_count, 
   float base_len);
double MidpointRiemann(double left_endpt, double right_endpt, double rect_count, 
   double base_len);

/* Function we're integrating */
double ex3_accel_double(double time);
float ex3_accel_float(float time);
double ex3_vel_double(double time);
float ex3_vel_float(float time);
double ex3_pos(double time);
double funct_to_integrate_double(double x); 
float funct_to_integrate_float(float x); 

int main(void) {
   int my_rank, comm_sz, n, local_n;   
   #ifdef PART2
      float a, b, local_a, local_b;
      const float step_size = .01;
      float local_int_area, total_int_area;
   #elif defined(PART3) || defined(PART4)
      double a, b, local_a, local_b;
      const double step_size = .001;
      double local_int_area, total_int_area;
   #endif

   /* Let the system do what it needs to start up MPI */
   MPI_Init(NULL, NULL);

   /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   a = 0.0;
   #ifdef HALF_TIME
      b = 900.0;
   #else
      b = 1800.0;
   #endif
   
   //Get_input(my_rank, comm_sz, &a, &b, &n);
   
   #ifdef PART2
      MPI_Bcast(&a, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   #elif defined(PART3) || defined(PART4)
      MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   #endif
   //MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);

   // step size in seconds (10 ms)
   // amounts of steps
   n = (b - a) / step_size;
   
   if(my_rank == 0) printf("my_rank=%d, a=%15.14lf, b=%15.14lf, number of total steps=%d\n", my_rank, a, b, n);

   local_n = n/comm_sz;  /* So is the number of quadratures  */

   /* Length of each process' interval of
    * integration = local_n*step_size.  So my interval
    * starts at: */
   local_a = a + my_rank*local_n*step_size;
   local_b = local_a + local_n*step_size;

   printf("my_rank=%d, start a=%lf, end b=%lf, number of quadratures = %d, step_size=%lf\n",
           my_rank, local_a, local_b, local_n, step_size);

   #ifdef PART2
      local_int_area = LeftRiemann(local_a, local_b, local_n, step_size);
   #elif defined(PART3)
      local_int_area = MidpointRiemann(local_a, local_b, local_n, step_size);
   #elif defined(PART4)
      local_int_area = Trap(local_a, local_b, local_n, step_size);
   #endif
   
   #ifdef PART2
      printf("After LeftRiemann: my_rank=%d, integrated area = %15.7lf, step_size %lf, number quadratures=%d\n", 
            my_rank, local_int_area, step_size, local_n);
   #elif defined(PART3)
      printf("After MidpointRiemann: my_rank=%d, integrated area = %20.15lf, step_size %lf, number quadratures=%d\n", 
         my_rank, local_int_area, step_size, local_n);
   #elif defined(PART4)
      printf("After Trap: my_rank=%d, integrated area = %20.15lf, step_size %lf, number quadratures=%d\n", 
         my_rank, local_int_area, step_size, local_n);
   #endif

   /* Add up the integrals calculated by each process */
   #ifdef PART2
      MPI_Reduce(&local_int_area, &total_int_area, 1, MPI_FLOAT, MPI_SUM, 0,
         MPI_COMM_WORLD);
   #elif defined(PART3) || defined(PART4)
      MPI_Reduce(&local_int_area, &total_int_area, 1, MPI_DOUBLE, MPI_SUM, 0,
         MPI_COMM_WORLD);
   #endif
   

   /* Print the result */
   if (my_rank == 0) {
      #ifdef PART2
         printf("After Reduce: with n = %d quadratures, our estimate\n", n);
         printf("of the integral from %lf to %lf = %15.7lf\n",
          a, b, total_int_area);
      #elif defined(PART3) || defined(PART4)
         printf("After Reduce: with n = %d quadratures, our estimate\n", n);
         printf("of the integral from %lf to %lf = %20.15lf\n",
            a, b, total_int_area);
      #endif
   }

   /* Shut down MPI */
   MPI_Finalize();

   return 0;
} /*  main  */

/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral 
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count 
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(
      double left_endpt  /* in */, 
      double right_endpt /* in */, 
      int    trap_count  /* in */, 
      double base_len    /* in */) {
   double estimate, x; 
   int i;

   estimate = (funct_to_integrate_double(left_endpt) + funct_to_integrate_double(right_endpt))/2.0;

   for (i = 1; i <= trap_count-1; i++) 
   {
      x = left_endpt + i*base_len;
      estimate += funct_to_integrate_double(x);
   }
   estimate = estimate*base_len;

   return estimate;
} /*  Trap  */


float LeftRiemann(
      float left_endpt, 
      float right_endpt, 
      int    rect_count, 
      float base_len) 
{
   float left_value, x, area=0.0; 
   int i;

   // estimate of function on left side to forward integrate
   x = left_endpt;
   left_value = funct_to_integrate_float(x);

   for (i = 1; i <= rect_count; i++) 
   {
      // add area of each rectangle to overall area sum
      area += left_value * base_len;

      // advance x by base length for new values to add to area
      x += base_len;
      left_value = funct_to_integrate_float(x);
   }

   return area;

} /*  LeftRiemann  */

double MidpointRiemann(
   double left_endpt, 
   double right_endpt, 
   double rect_count, 
   double base_len
   ) 
{
   double x, midpoint_value, area=0.0;
   int i;

   x = left_endpt + base_len / 2;
   midpoint_value = funct_to_integrate_double(x);

   for (i = 1; i <= rect_count; i++) 
   {
      // add area of each rectangle to overall area sum
      area += midpoint_value * base_len;

      // advance x by base length for new values to add to area
      x += base_len;
      midpoint_value = funct_to_integrate_double(x);
   }

   return area;
}


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double funct_to_integrate_double(double x) 
{
   #ifdef VELOCITY
      return(ex3_accel_double(x));
   #elif defined(DISPLACEMENT)
      return(ex3_vel_double(x));  
   #endif
}

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
float funct_to_integrate_float(float x) 
{
   #ifdef VELOCITY
      return(ex3_accel_float(x));
   #elif defined(DISPLACEMENT)
      return(ex3_vel_float(x));  
   #endif
}


double ex3_accel_double(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that acceleration will peak to result in translation of 122,000.0 meters
    //static double ascale=0.2365893166123;
    static double ascale=0.236589076381454;

    return (sin(time/tscale)*ascale);
}

float ex3_accel_float(float time)
{
    // computation of time scale for 1800 seconds
    static float tscale=1800.0/(2.0*M_PI);
    // determined such that acceleration will peak to result in translation of 122,000.0 meters
    //static double ascale=0.2365893166123;
    static float ascale=0.236589076381454;

    return (sinf(time/tscale)*ascale);
}

// determined based on known anti-derivative of ex4_accel function
double ex3_vel_double(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.236589076381454*1800.0/(2.0*M_PI);

    return ((-cos(time/tscale)+1)*vscale);
}

// determined based on known anti-derivative of ex4_accel function
float ex3_vel_float(float time)
{
    // computation of time scale for 1800 seconds
    static float tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static float vscale=0.236589076381454*1800.0/(2.0*M_PI);

    return ((-cos(time/tscale)+1)*vscale);
}


// determined based on known anti-derivative of ex4_vel function
double ex3_pos(double time)
{
    // computation of time scale for 1800 seconds
    static double tscale=1800.0/(2.0*M_PI);
    // determined such that velocity will peak to result in translation of 122,000.0 meters
    static double vscale=0.236589076381454*1800.0/(2.0*M_PI);

    return ((-tscale*(sin(time/tscale)+time))*vscale);
}
