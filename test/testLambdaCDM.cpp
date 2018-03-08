#include <iostream>
#include <string>
#include <random>
#include <map>
#include <vector>
#include "gmock/gmock.h"
#include "LambdaCDM.hpp"


TEST(LambdaCDM_model, test_1)
{

 std::map<std::string, double> params = {{"Omega_m0", 0.3}};
 LambdaCDM_model LCDM_model_ins;

 try {
LCDM_model_ins.setOmega_m0(params.at("Omega_m0"));
LCDM_model_ins.setOmega_DE0(0.7);
LCDM_model_ins.setOmega_b0(0.04);
 }
 catch (const std::out_of_range& oor) {
std::cerr << "Out of Range error in std::map params:" <<
    " some parameter is not set " << oor.what() << '\n';
std::exit(1);
 }

 LCDM_model_ins.setOmega_curv0();

 double Omega_b0_value = LCDM_model_ins.getOmega_b0();
 double Omega_b0_value_exp = 0.04;
 ASSERT_EQ(Omega_b0_value_exp, Omega_b0_value);

 double Omega_m0_value = LCDM_model_ins.getOmega_m0();
 double Omega_m0_value_exp = params.at("Omega_m0");
 ASSERT_EQ(Omega_m0_value_exp, Omega_m0_value);

 double Omega_DE0_value = LCDM_model_ins.getOmega_DE0();
 double Omega_DE0_value_exp = 0.7;
 ASSERT_EQ(Omega_DE0_value_exp, Omega_DE0_value);

 double abs_err = 1.0e-3;

 double Omega_curv0_value = LCDM_model_ins.getOmega_curv0();
 double Omega_curv0_value_exp = 0.0;
 ASSERT_NEAR(Omega_curv0_value_exp, Omega_curv0_value, abs_err);

 double atoH_value = LCDM_model_ins.atoH(0.5);
 double atoH_value_exp = 1.76097;
 ASSERT_NEAR(atoH_value_exp, atoH_value, abs_err);

 // std::cout << LCDM_model_ins.getOmega_b0() << "\n";
 // std::cout << LCDM_model_ins.getOmega_m0() << "\n";
 // std::cout << LCDM_model_ins.getOmega_DE0() << "\n";
 // std::cout << LCDM_model_ins.getOmega_curv0() << "\n";
 // std::cout << LCDM_model_ins.atoH(0.5) << "\n";

}

TEST(LambdaCDM_model, test_2)
{

 LambdaCDM_model LCDM_model_ins;

 std::size_t z_sn_size_value = LCDM_model_ins.get_z_sn_size();
 std::size_t z_sn_size_value_exp = 0;
 ASSERT_EQ(z_sn_size_value_exp, z_sn_size_value);

 LCDM_model_ins.setSN_data();

 z_sn_size_value = LCDM_model_ins.get_z_sn_size();
 z_sn_size_value_exp = 580;
 ASSERT_EQ(z_sn_size_value_exp, z_sn_size_value);

}

TEST(LambdaCDM_model, test_3)
{

    const std::map<std::string, double> params = {{"Omega_m0", 0.3}};
    LambdaCDM_model LCDM_model_ins;
    const double hub = 0.7;
    const double Omega_b0_default = 0.022765 * std::pow(hub, -2.0);
    
    try {
	LCDM_model_ins.setOmega_m0(params.at("Omega_m0"));
	LCDM_model_ins.setOmega_DE0(0.7);
	LCDM_model_ins.setOmega_b0(Omega_b0_default);
    }
    catch (const std::out_of_range& oor) {
	std::cerr << "Out of Range error in std::map params:" <<
	    " some parameter is not set " << oor.what() << '\n';
	std::exit(1);
    }
    
    LCDM_model_ins.setOmega_curv0();
    LCDM_model_ins.setSN_data();
    LCDM_model_ins.chisq_SN_init();
    LCDM_model_ins.precompute_atochi_spline_table();
    LCDM_model_ins.chisq_BAO_CMB_init();
    LCDM_model_ins.precompute_ators_spline_table();
    LCDM_model_ins.precompute_atoR_spline_table();

    double abs_err = 1.0e-3;
    
    double atoH_value = LCDM_model_ins.atoH(0.5);
    double atoH_value_exp = 1.76097;
    ASSERT_NEAR(atoH_value_exp, atoH_value, abs_err);
    
    double atodL_value = LCDM_model_ins.atodL(0.5);
    double atodL_value_exp = 6607.33;
    abs_err = 1.0e-2;
    ASSERT_NEAR(atodL_value_exp, atodL_value, abs_err);
    
    double chisq_SN_value = LCDM_model_ins.chisq_SN();
    double chisq_SN_value_exp = 563.514;
    ASSERT_NEAR(chisq_SN_value_exp, chisq_SN_value, abs_err);

    double z_star_value = LCDM_model_ins.z_star();
    double z_star_value_exp = 1091.68;
    ASSERT_NEAR(z_star_value_exp, z_star_value, abs_err);
    
    double atoR_value = LCDM_model_ins.atoR(1 / (1 + LCDM_model_ins.z_star()) );
    double atoR_value_exp = 1.74169;
    ASSERT_NEAR(atoR_value_exp, atoR_value, abs_err);

    
    double a_star_value = 1 / (1 + LCDM_model_ins.z_star());
    double a_star_value_exp = 0.000915179;
    ASSERT_NEAR(a_star_value_exp, a_star_value, abs_err);
    
    std::cout << "ators(1 / (1 + z_star) ) = " << "\n";
    std::cout << LCDM_model_ins.ators(1 / (1 + LCDM_model_ins.z_star()) ) << "\n";
    double ators_value = LCDM_model_ins.ators(1 / (1 + LCDM_model_ins.z_star()) );
    double ators_value_exp = 148.89;
    ASSERT_NEAR(ators_value_exp, ators_value, abs_err);

    std::cout << LCDM_model_ins.atolA(1 / (1 + LCDM_model_ins.z_star()) ) << "\n";
    double atolA_value = LCDM_model_ins.atolA(1 / (1 + LCDM_model_ins.z_star()) );
    double atolA_value_exp = 287.278;
    ASSERT_NEAR(atolA_value_exp, atolA_value, abs_err);

    double chisq_WMAP7_CMB_BAO_value = LCDM_model_ins.chisq_WMAP7_CMB_BAO();
    double chisq_WMAP7_CMB_BAO_value_exp = 508.33;
    ASSERT_NEAR(chisq_WMAP7_CMB_BAO_value_exp, chisq_WMAP7_CMB_BAO_value, abs_err);
    
}

// bad values
TEST(LambdaCDM_model, test_4)
{

    const std::map<std::string, double> params = {{"Omega_m0", 0.2}};
    LambdaCDM_model LCDM_model_ins;
    const double hub = 0.7;
    const double Omega_b0_default = 0.022765 * std::pow(hub, -2.0);
    
    try {
	LCDM_model_ins.setOmega_m0(params.at("Omega_m0"));
	LCDM_model_ins.setOmega_DE0(1.7);
	LCDM_model_ins.setOmega_b0(Omega_b0_default);
    }
    catch (const std::out_of_range& oor) {
	std::cerr << "Out of Range error in std::map params:" <<
	    " some parameter is not set " << oor.what() << '\n';
	std::exit(1);
    }
    
    LCDM_model_ins.setOmega_curv0();
    LCDM_model_ins.setSN_data();
    LCDM_model_ins.chisq_SN_init();
    LCDM_model_ins.precompute_atochi_spline_table();
    LCDM_model_ins.chisq_BAO_CMB_init();
    LCDM_model_ins.precompute_ators_spline_table();
    LCDM_model_ins.precompute_atoR_spline_table();
    
    std::cout << LCDM_model_ins.chisq_SN() << "\n";
    double chisq_SN_value = LCDM_model_ins.chisq_SN();
    double chisq_SN_value_exp = 1.0e30;
    ASSERT_EQ(chisq_SN_value_exp, chisq_SN_value);

    std::cout << LCDM_model_ins.chisq_WMAP7_CMB_BAO() << "\n";
    double chisq_WMAP7_CMB_BAO_value = LCDM_model_ins.chisq_WMAP7_CMB_BAO();
    double chisq_WMAP7_CMB_BAO_value_exp = 1.0e30;
    ASSERT_EQ(chisq_WMAP7_CMB_BAO_value_exp, chisq_WMAP7_CMB_BAO_value);
    
}

double chisq_GC()
{
    double z_GC_1 = 0.35;

    static double atoH_value = 0.0;

    double chisq_GC_1_value = 0.0, chisq_GC_2_value = 0.0;

    arma::vec fit_vector_1 = {0.5 * 0.5, 0.7 / 0.4};
   
    arma::mat Norm_Cov_mat_GC_1 = {1.0000, 0.0604, 0.0604, 1.0000};
    Norm_Cov_mat_GC_1.reshape(2, 2);
    arma::vec sigma_vec_GC_1 = {0.0018, 0.26};

    arma::mat Cov_mat_GC_1 = arma::zeros<arma::mat>(2, 2);

    for(unsigned int i = 0; i < Cov_mat_GC_1.n_rows; i++){
	for(unsigned int j = 0; j < Cov_mat_GC_1.n_cols; j++){
            Cov_mat_GC_1(i, j) = sigma_vec_GC_1(i) * sigma_vec_GC_1(j) * Norm_Cov_mat_GC_1(i, j);
	} 
    }

    std::cout << "Cov_mat_GC_1" << "\n";
    std::cout  << Cov_mat_GC_1 << "\n";

    arma::mat Cov_mat_GC_1_inv = Cov_mat_GC_1.i();
    arma::vec fit_vector_GC_1 = {0.0434, 6.60};

    arma::vec cchi_GC_1_intm = arma::zeros<arma::vec>(2);

    for (unsigned int i = 0; i < Cov_mat_GC_1_inv.n_rows; ++i)
	{
	    cchi_GC_1_intm(i) = arma::dot(Cov_mat_GC_1_inv.row(i), fit_vector_1 - fit_vector_GC_1);
	}
    chisq_GC_1_value = arma::dot(fit_vector_1 - fit_vector_GC_1, cchi_GC_1_intm);

    double z_GC_2 = 0.57;

    arma::vec fit_vector_2 = {0.7 * 0.7, 0.9 / 0.6};
    
    arma::mat Norm_Cov_mat_GC_2 = {1.0000, 0.0604, 0.0604, 1.0000};
    Norm_Cov_mat_GC_2.reshape(2, 2);
    arma::vec sigma_vec_GC_2 = {0.0031, 0.27};

    arma::mat Cov_mat_GC_2 = arma::zeros<arma::mat>(2, 2);

    for(unsigned int i = 0; i < Cov_mat_GC_2.n_rows; i++){
	for(unsigned int j = 0; j < Cov_mat_GC_2.n_cols; j++){
            Cov_mat_GC_2(i, j) = sigma_vec_GC_2(i) * sigma_vec_GC_2(j) * Norm_Cov_mat_GC_2(i, j);
	} 
    }

    std::cout << "Cov_mat_GC_2" << "\n";
    std::cout  << Cov_mat_GC_2 << "\n";

    arma::mat Cov_mat_GC_2_inv = Cov_mat_GC_2.i();
    arma::vec fit_vector_GC_2 = {0.0484, 8.95};

    arma::vec cchi_GC_2_intm = arma::zeros<arma::vec>(2);

    for (unsigned int i = 0; i < Cov_mat_GC_2_inv.n_rows; ++i)
	{
	    cchi_GC_2_intm(i) = arma::dot(Cov_mat_GC_2_inv.row(i), fit_vector_2 - fit_vector_GC_2);
	}
    chisq_GC_2_value = arma::dot(fit_vector_2 - fit_vector_GC_2, cchi_GC_2_intm);

    double chisq_GC_value = chisq_GC_1_value + chisq_GC_2_value;
    
    return chisq_GC_value;
}


TEST(LambdaCDM_model, test_chisq_GC)
{

    double chiqs_GC_exp = 35438.458;
    double chiqs_GC_value = chisq_GC();
    double abs_err = 1.0e-3;
    std::cout << chiqs_GC_value << "\n";
    ASSERT_NEAR(chiqs_GC_value, chiqs_GC_exp, abs_err);

}

