#include "LambdaCDM.hpp"

double LambdaCDM_model::atoH(double a){

    double atoH_value = getOmega_m0() * std::pow(a, -3) + Omega_r0 * std::pow(a, -4) +
    				  getOmega_DE0() + getOmega_curv0() * std::pow(a, -2);

    if (atoH_value < 0)
    	throw std::invalid_argument("get complex hubble parameter");

    return std::sqrt(atoH_value);
}
    
