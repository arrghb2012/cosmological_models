#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <gsl/gsl_spline.h>
#include "H5Cpp.h"
#include <memory>
#include <vector>

class Spline
{ 
  public:
    Spline() {};
    void alloc_Spline (int num_pts, std::string type);
    void setup_Spline (double* x_array, double* y_array, int num_pts);
    ~Spline ();

    double y (const double x);
    double yp (const double x);
    double ypp (const double x);
    double y_integrate (const double a, const double b);

  private:    
    std::string spline_type;
    gsl_interp_accel *accel_ptr;
    gsl_spline *spline_ptr;

};

typedef std::shared_ptr<H5::H5File> H5FilePtr;

H5FilePtr create_or_open(const H5std_string& fname);

int write_array(const std::string& filename, const std::string& datasetname, const std::vector<double>& vec);

#endif
