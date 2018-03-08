#ifndef COSMOLOGICALMODEL_HPP
#define COSMOLOGICALMODEL_HPP

#include "utils.hpp"
#include <string>
#include <armadillo>

class CosmologicalModel
{
public:

    ~CosmologicalModel() {};

    virtual double atoH(double a);

    virtual void setOmega_b0(double Omega_b0);
    virtual double getOmega_b0() const;

    virtual void setOmega_m0(double Omega_m0);
    virtual double getOmega_m0() const;

    virtual void setOmega_DE0(double Omega_DE0);
    virtual double getOmega_DE0() const;

    virtual void setOmega_curv0();
    virtual double getOmega_curv0() const;

    void chisq_SN_init();

    void setSN_data();
    std::size_t get_z_sn_size();
    
    void precompute_atochi_spline_table();
    double atodL(double a);
    double atomu(double a);

    double chisq_SN();

    void chisq_BAO_CMB_init();

    double atoDV(double a);

    double atodrs(double a);

    void precompute_ators_spline_table();

    double ators(double a);

    void precompute_atoR_spline_table();
    double atoR(double a);
    
    double z_star();
    double z_drag();

    double atolA(double a);
    double atoDA(double a);

    double chisq_WMAP7_CMB_BAO();
    double chisq_Planck();
    double chisq_WMAP9();
    double chisq_GC();

protected:
    double mOmega_b0;
    double mOmega_m0;
    double mOmega_DE0;
    double mOmega_curv0;
    const double hub = 0.7;
    const double H_0 = hub / 2998;
    const double N_eff = 3.04;
    const double Omega_gamma0 = 2.469e-5 * std::pow(hub, -2.0);
    const double Omega_r0 = Omega_gamma0 * (1 + 0.2271 * N_eff);
    
private:
    std::vector<double> z_sn, mu_exp, mu_err;
    Spline cubic_spline_for_atodchi;
    Spline cubic_spline_atodrs;
    Spline cubic_spline_for_atoR;

    double chisq_SN_value = 0.0;
    double chisq_WMAP7_CMB_BAO_value = 0.0;
    double chisq_Planck_value = 0.0;
    double chisq_WMAP9_value = 0.0;
    double chisq_GC_value = 0.0;

    arma::vec a_array_for_chisq_SN;
    unsigned int a_array_size_for_chisq_SN;
    arma::vec atodchi_array;

    arma::vec a_array_atodrs;
    unsigned int a_array_atodrs_size;
    arma::vec atodrs_array;

    arma::vec a_array_atodR;
    unsigned int a_array_atodR_size;
    arma::vec atodR_array;

};


#endif 
