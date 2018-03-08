//  file: GslSpline.cpp
// 
//  Definitions for the GslSpline C++ class. 
//
//  Programmer:  Dick Furnstahl  furnstahl.1@osu.edu
//
//  Revision history:
//     01/25/09  Original version using gsl_cubic_spline_test.cpp as a guide.
//
//*****************************************************************
// include files
#include <iostream>
#include <string>     // C++ strings                                 
#include <stdlib.h>      // this has exit
#include <gsl/gsl_errno.h>	// header for gsl error handling
#include <gsl/gsl_spline.h>	// header for gsl splining routines 
#include "GslSpline.hpp"      // include the header for this class

//********************************************************************

// Constructor for Spline
Spline::Spline(double* x_array, double* y_array, int num_pts, std::string type) 
{
  spline_type = type;    // set the private variable for the spline type
  
  // Allocate the accelerator
  accel_ptr = gsl_interp_accel_alloc ();
 
  // Allocate the spline according to spline_type
  if (spline_type == "cubic")
  {
    spline_ptr = gsl_spline_alloc (gsl_interp_cspline, num_pts);
  }
  else if (spline_type == "linear")
  {
    spline_ptr = gsl_spline_alloc (gsl_interp_linear, num_pts);
  }
  else
  {
    std::cout << "Illegal spline type!" << std::endl;
    exit (1);  // time to quit!
  }

  // Initialize the spline 
  gsl_spline_init (spline_ptr, x_array, y_array, num_pts);  
}

Spline::~Spline () // Destructor for Spline
{
  // Free the accelerator and spline object 
  gsl_spline_free (spline_ptr);
  gsl_interp_accel_free (accel_ptr);
}

double Spline::y (const double x)  // find y(x)
{
  return gsl_spline_eval (spline_ptr, x, accel_ptr);
}

double Spline::yp (const double x)  // find y'(x)
{
  return gsl_spline_eval_deriv (spline_ptr, x, accel_ptr);
}

double Spline::ypp (const double x)  // find y''(x)
{
  return gsl_spline_eval_deriv2 (spline_ptr, x, accel_ptr);
}

void mySpline::setup_mySpline(double* x_array, double* y_array, int num_pts, std::string type) 
{
  spline_type = type;    // set the private variable for the spline type
  
  // Allocate the accelerator
  accel_ptr = gsl_interp_accel_alloc ();
 
  // Allocate the spline according to spline_type
  if (spline_type == "cubic")
  {
    spline_ptr = gsl_spline_alloc (gsl_interp_cspline, num_pts);
  }
  else if (spline_type == "linear")
  {
    spline_ptr = gsl_spline_alloc (gsl_interp_linear, num_pts);
  }
  else
  {
    std::cout << "Illegal spline type!" << std::endl;
    exit (1);  // time to quit!
  }

  // Initialize the spline 
  gsl_spline_init (spline_ptr, x_array, y_array, num_pts);  
}

mySpline::~mySpline () // Destructor for Spline
{
  // Free the accelerator and spline object 
  gsl_spline_free (spline_ptr);
  gsl_interp_accel_free (accel_ptr);
}

double mySpline::y (const double x)  // find y(x)
{
  return gsl_spline_eval (spline_ptr, x, accel_ptr);
}

double mySpline::yp (const double x)  // find y'(x)
{
  return gsl_spline_eval_deriv (spline_ptr, x, accel_ptr);
}

double mySpline::ypp (const double x)  // find y''(x)
{
  return gsl_spline_eval_deriv2 (spline_ptr, x, accel_ptr);
}

double mySpline::y_integrate (const double a, const double b)  // find y''(x)
{
    return gsl_spline_eval_integ (spline_ptr, a, b, accel_ptr);
}


//********************************************************************
