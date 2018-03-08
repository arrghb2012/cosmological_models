#include <iostream>
#include <string>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include "utils.hpp"

void Spline::alloc_Spline(int num_pts, std::string type) 
{
  spline_type = type;
  
  accel_ptr = gsl_interp_accel_alloc ();
 
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
    exit (1);
  }

}

void Spline::setup_Spline(double* x_array, double* y_array, int num_pts) 
{
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

double Spline::y_integrate (const double a, const double b)  // find y''(x)
{
    return gsl_spline_eval_integ (spline_ptr, a, b, accel_ptr);
}

typedef std::shared_ptr<H5::H5File> H5FilePtr;

H5FilePtr create_or_open(const H5std_string& fname)
{
    H5::Exception::dontPrint();
    H5::H5File* file = 0;

    try {
        file = new H5::H5File(fname, H5F_ACC_RDWR);
    } catch(const H5::FileIException&) {
        file = new H5::H5File(fname, H5F_ACC_TRUNC);
    }

    return H5FilePtr(file);
}

int write_array(const std::string& filename, const std::string& datasetname, const std::vector<double>& vec)
{
    const H5std_string	FILE_NAME( filename.c_str() );
    const H5std_string	DATASET_NAME( datasetname.c_str() );
    int NX = vec.size();
    const int 	RANK = 1;


    // Try block to detect exceptions raised by any of the calls inside it
    
    try
	{
	    H5::Exception::dontPrint();

	    H5FilePtr file = create_or_open( FILE_NAME );

	    hsize_t     dimsf[1];              // dataset dimensions
	    dimsf[0] = NX;
	    H5::DataSpace dataspace( RANK, dimsf );

	    H5::FloatType datatype( H5::PredType::NATIVE_DOUBLE );
	    datatype.setOrder( H5T_ORDER_LE );

	    H5::DataSet dataset = file->createDataSet( DATASET_NAME, datatype, dataspace );

	    dataset.write( &vec[0], H5::PredType::NATIVE_DOUBLE );
	}  // end of try block

    catch( H5::FileIException error )
	{
	    error.printError();
	    return -1;
	}

    catch( H5::DataSetIException error )
	{
	    error.printError();
	    return -1;
	}

    catch( H5::DataSpaceIException error )
	{
	    error.printError();
	    return -1;
	}

    catch( H5::DataTypeIException error )
	{
	    error.printError();
	    return -1;
	}

    return 0;  // successfully terminated
}


//********************************************************************
