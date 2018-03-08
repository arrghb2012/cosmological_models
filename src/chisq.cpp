#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <time.h>
#include <string>
#include <array>
#include <gsl/gsl_integration.h>
#include "mcmc.h"

std::vector<double> z_sn, mu_exp, mu_err;
const double hub = 0.7;
const double H_0 = hub / 2998;
const double N_eff = 3.04;
const double Omega_gamma = 2.469e-5 * pow(hub, -2.0);
const double Omega_r = Omega_gamma * (1 + 0.2271 * N_eff);
const double Omega_b = 0.022765 * pow(hub, -2.0);

void get_SN_data(){
    double z_sn_temp, mu_exp_temp, mu_err_temp;
    const char* input_file_name = "../SCPUnion2.1_mu_vs_z_modified.txt";
    std::ifstream my_in(input_file_name);
    if (my_in.fail()) {
	std::cerr << "Unable to open the file " << input_file_name
		  << " for input." << std::endl;
    }

    while (my_in >> z_sn_temp >> mu_exp_temp >> mu_err_temp) {
    	z_sn.push_back(z_sn_temp);
    	mu_exp.push_back(mu_exp_temp);
    	mu_err.push_back(mu_err_temp);
    }
}

struct cosm_params { 
    double Omega_m; 
    double Omega_Lambda; 
    double tilde_B;
};

struct params_ind { 
    const unsigned int Omega_m_idx = 0; 
    const unsigned int Omega_Lambda_idx = 1; 
    const unsigned int tilde_B_idx = 2; 
};

double atoH(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    double atoH = sqrt(c_p->Omega_Lambda - 6 * c_p->tilde_B*log(a) + c_p->Omega_m * pow(a, -3) + Omega_r * pow(a, -4));
    return atoH;
}

double atodchi(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    return 1.0 / (pow(a, 2.0) * atoH(a, c_p));
}

double atodL_1(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    double atochi, d_L;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    double b = 1.0;
    double epsabs = 1.49e-8;
    double epsrel = 1.49e-8;
    double result, error;

    cosm_params c_params = {c_p->Omega_m, c_p->Omega_Lambda, c_p->tilde_B};
    gsl_function F;
    F.function = &atodchi;
    F.params = &c_params;

    gsl_integration_qags( & F, a, b, epsabs, epsrel, 1000, w, &result, &error);
    gsl_integration_workspace_free(w);
    
    atochi = result;
    double Omega_curv = 0.0;
    if (Omega_curv > 0){
	d_L = 1 / H_0 / (a * sqrt(fabs(Omega_curv))) * sinh(sqrt(fabs(Omega_curv)) * atochi);
    }
    else if (Omega_curv < 0) {
	d_L = 1 / H_0 / (a * sqrt(fabs(Omega_curv))) * sin(sqrt(fabs(Omega_curv)) * atochi);
    }
    else  d_L = 1 / H_0 / a * atochi;
    return d_L;
}

double atomu_1(const arma::vec& params, double a){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return 5 * log10(atodL_1(a, &c_params)) + 25;
}

double get_atoH(const arma::vec& params, double a){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return atoH(a, &c_params);
}

double atoDV(const arma::vec& params, double a){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return pow(pow(a, 2.0) * pow(atodL_1(a, &c_params), 2.0) * (1.0 / a - 1.0) / atoH(a, &c_params), 1. / 3);
}

double atodrs(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    return 1.0 / H_0 / (pow(a, 2.0) * atoH(a, c_p) * sqrt(1 + (3 * Omega_b/ 4.0 / Omega_r) * a)) / sqrt(3.0);
}

double get_atodrs(const arma::vec& params, double a){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return atodrs(a, &c_params);
}

double ators(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    double ators;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    double epsabs = 1.49e-8;
    double epsrel = 1.49e-8;
    double result, error;

    cosm_params c_params = {c_p->Omega_m, c_p->Omega_Lambda, c_p->tilde_B};
    gsl_function F;
    F.function = &atodrs;
    F.params = &c_params;

    gsl_integration_qags( & F, 0.0, a, epsabs, epsrel, 1000, w, &result, &error);
    gsl_integration_workspace_free(w);
    
    ators = result;
    return ators;
}

double get_ators(const arma::vec& params, double a){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return ators(a, &c_params);
}

double atolA(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    return M_PI * atodL_1(a, c_p) * a / ators(a, c_p);
}

double get_atolA(const arma::vec& params, double a){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return atolA(a, &c_params);
}

double atoR(double a, void * params){
    struct cosm_params * c_p = (struct cosm_params *)params;
    double atochi, R;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    double b = 1.0;
    double epsabs = 1.49e-8;
    double epsrel = 1.49e-8;
    double result, error;

    cosm_params c_params = {c_p->Omega_m, c_p->Omega_Lambda, c_p->tilde_B};
    gsl_function F;
    F.function = &atodchi;
    F.params = &c_params;

    gsl_integration_qags( & F, a, b, epsabs, epsrel, 1000, w, &result, &error);
    gsl_integration_workspace_free(w);
    
    atochi = result;
    double Omega_curv = 0.0;
    // double H_0 = 0.7 / 2998;
    if (Omega_curv > 0){
	R = sqrt(c_p->Omega_m)/(sqrt(fabs(Omega_curv))) * sinh(sqrt(fabs(Omega_curv)) * atochi);
    }
    else if (Omega_curv < 0) {
	R = sqrt(c_p->Omega_m)/(sqrt(fabs(Omega_curv))) * sin(sqrt(fabs(Omega_curv)) * atochi);
    }
    else  R = sqrt(c_p->Omega_m) * atochi;
    return R;
}

double z_star(void * params){
    double g1, g2, z_star;
    struct cosm_params * c_p = (struct cosm_params *)params;
    g1 = (0.0783 * pow(Omega_b * pow(hub, 2), -0.238))/(1 + 39.5 * pow(Omega_b * pow(hub, 2), 0.763));
    g2 = 0.560 / (1 + 21.1 * pow(Omega_b * pow(hub, 2), 1.81));
    z_star = 1048 * (1 + 0.00124 * pow(Omega_b * pow(hub, 2), -0.738)) * (1 + g1 * pow(c_p->Omega_m * pow(hub, 2), g2));
    return z_star;
}

double get_z_star(const arma::vec& params){
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    return z_star(&c_params);
}

bool NOB(double Omega_m, double Omega_Lambda){
    double NOBTEST;
    bool NOB = false;

    if (Omega_m == 0.0)
    {
	NOBTEST = 1.0;
    }
    else if (Omega_m <= 0.5)
    {
	NOBTEST = 4.0 * Omega_m * pow(cosh(acosh((1 - Omega_m) / Omega_m) / 3.0), 3.0); 
    }
    else {
	NOBTEST = 4.0 * Omega_m * pow(cosh(acos((1 - Omega_m) / Omega_m) / 3.0), 3.0); 
    }

    if (NOBTEST > Omega_Lambda + 0.02)
    {
	NOB = true;
    }
    
    return NOB;
}

double chisq_SN(const arma::vec& params)
{
    double aA = 0.0, aB = 0.0, Csn = 0.0, cchi_SN = 0.0;

    params_ind pi; 
    double Omega_m = params(pi.Omega_m_idx) ;
    double Omega_Lambda = params(pi.Omega_Lambda_idx);
    double atomu_temp = 0.0, denom_temp = 0.0;

    if (NOB(Omega_m, Omega_Lambda))
    {
	for (unsigned int i = 0; i < z_sn.size(); ++i)
	{
	    denom_temp = mu_err[i] * mu_err[i];
	    atomu_temp = mu_exp[i] - atomu_1(params, 1/(1+z_sn[i]));
	    Csn += 1 / denom_temp;
	    aA += atomu_temp * atomu_temp / denom_temp;
	    aB += atomu_temp / denom_temp;
	}
	cchi_SN = aA - aB * aB / Csn;
    }
    else
    {
	cchi_SN = 1.0e30;
    }
    return cchi_SN;

}

double chisq_joint(const arma::vec& params)
{
    params_ind pi; 
    cosm_params c_params = {params(pi.Omega_m_idx), params(pi.Omega_Lambda_idx), params(pi.tilde_B_idx)};
    double z_star_value = z_star(&c_params);
    arma::vec CMBfit_vector = {atolA(1 / ( z_star_value + 1 ), &c_params), atoR(1 / ( z_star_value + 1 ), &c_params), 
			       z_star_value};
    arma::mat CCMB = {2.305, 29.698, -1.333, 29.698, 6825.27, -113.180, -1.333, -113.180, 3.414};
    CCMB.reshape(3, 3);
    arma::vec CMBfit_vector_WMAP = {302.09, 1.725, 1091.3};
    double cchi_CMB, cchi_BAO;
    arma::vec cchi_CMB_intm = arma::zeros<arma::vec>(3);
    for (unsigned int i = 0; i < CCMB.n_rows; ++i)
    {
	cchi_CMB_intm(i) = arma::dot(CCMB.row(i), CMBfit_vector - CMBfit_vector_WMAP);
    }
    cchi_CMB = arma::dot(CMBfit_vector - CMBfit_vector_WMAP, cchi_CMB_intm);

    cchi_BAO = pow((atoDV(params, 1 / ( 1 + 0.35 )) / atoDV(params, 1 / ( 1 + 0.2 )) - 1.736 ), 2) / pow(0.065, 2);
    double cchi_joint = chisq_SN(params) + cchi_CMB + cchi_BAO;
    return cchi_joint;

}

int main()
{
 auto start = std::chrono::system_clock::now();
 get_SN_data();
 arma::vec av_lower_bound = {0.0, 0.0, -0.2};
 arma::vec av_upper_bound = {1.0, 1.0, 0.2};
 arma::vec av_dpar = {0.02, 0.02, 0.002};
 unsigned long av_NUMBER_OF_STEPS = 100000;
 int av_NCHAINS = 8;
 double av_mh_factor = 1.0;
 MCMC MCMC_ins(av_lower_bound, av_upper_bound, av_dpar, av_NUMBER_OF_STEPS,
          av_NCHAINS, av_mh_factor);
 MCMC_ins.do_mcmc(chisq_joint, "chains.dat");
 auto end = std::chrono::system_clock::now();
 auto diff = end - start;
 std::cout << std::chrono::duration < double > (diff).count() << "s" << std::endl;

}


