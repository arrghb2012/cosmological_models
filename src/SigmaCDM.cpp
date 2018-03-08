#include "SigmaCDM.hpp"

double SigmaCDM_model::atoH(double a){

    double atoH_value = getOmega_m0() * std::pow(a, -3) + Omega_r0 * std::pow(a, -4) +
	getOmega_DE0() - get_tildeA() * std::log(a) + get_tildeB() * std::pow(a, -get_nu()) + getOmega_curv0() * std::pow(a, -2);

    if (atoH_value < 0)
    	throw std::invalid_argument("get complex hubble parameter");

    return std::sqrt(atoH_value);
}


void SigmaCDM_model::setOmega_curv0(){
    mOmega_curv0 = 1 - get_tildeB() - getOmega_m0() - Omega_r0 - getOmega_DE0();
}

void SigmaCDM_model::set_tildeA(double tildeA)
{
    mtildeA = tildeA;
}

void SigmaCDM_model::set_tildeB(double tildeB)
{
    mtildeB = tildeB;
}

void SigmaCDM_model::set_nu(double nu)
{
    mnu = nu;
}

double SigmaCDM_model::get_tildeA()
{
    return mtildeA;
}

double SigmaCDM_model::get_tildeB()
{
    return mtildeB;
}

double SigmaCDM_model::get_nu()
{
    return mnu;
}

