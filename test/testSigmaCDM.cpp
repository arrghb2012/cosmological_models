#include <iostream>
#include <string>
#include <map>
#include <vector>
#include "gmock/gmock.h" 
#include "SigmaCDM.hpp"

TEST(SigmaCDM_model, test_1)
{
    
    std::map<std::string, double> cosm_params =
	{{"Omega_DEi", -19.553}, 
	 {"Omega_mi", -2.22098}};

    cosm_params["mu"] = 1.0;
    cosm_params["lam"] = 0.1;
    cosm_params["omegaDE_i"] = 0.0;

    SM_output SM_output_ins(cosm_params);
    SM_output_ins.solve_background();
    
    const double hub = 0.7;
    const double Omega_b0_default = 0.022765 * std::pow(hub, -2.0);
    
    SM_output_ins.setOmega_b0(Omega_b0_default);

    SM_output_ins.setOmega_curv0();

    SM_output_ins.setSN_data();
    SM_output_ins.chisq_SN_init();
    SM_output_ins.precompute_atochi_spline_table();
    SM_output_ins.chisq_BAO_CMB_init();
    SM_output_ins.precompute_ators_spline_table();
    SM_output_ins.precompute_atoR_spline_table();

    double abs_err = 1.0e-2;
    
     std::cout << SM_output_ins.atoH(0.5) << "\n";
    double atoH_value = SM_output_ins.atoH(0.5);
    double atoH_value_exp = 1.76097;
     ASSERT_EQ(expected, actual);
    EXPECT_NEAR(atoH_value_exp, atoH_value, abs_err);
    
     std::cout << SM_output_ins.atodL(0.5) << "\n";
    double atodL_value = SM_output_ins.atodL(0.5);
    double atodL_value_exp = 6607.33;
    abs_err = 1.0e-2;
    EXPECT_NEAR(atodL_value_exp, atodL_value, abs_err);

    std::cout << SM_output_ins.chisq_SN() << "\n";
    double chisq_SN_value = SM_output_ins.chisq_SN();
    double chisq_SN_value_exp = 563.514;
    EXPECT_NEAR(chisq_SN_value_exp, chisq_SN_value, abs_err);
    
    std::cout << SM_output_ins.z_star() << "\n";
    double z_star_value = SM_output_ins.z_star();
    double z_star_value_exp = 1091.68;
    EXPECT_NEAR(z_star_value_exp, z_star_value, abs_err);
    
    std::cout << SM_output_ins.atoR(1 / (1 + SM_output_ins.z_star()) ) << "\n";
    double atoR_value = SM_output_ins.atoR(1 / (1 + SM_output_ins.z_star()) );
    double atoR_value_exp = 1.74169;
    EXPECT_NEAR(atoR_value_exp, atoR_value, abs_err);

    double a_star_value = 1 / (1 + SM_output_ins.z_star());
    double a_star_value_exp = 0.000915179;
    EXPECT_NEAR(a_star_value_exp, a_star_value, abs_err);
    
    double ators_value = SM_output_ins.ators(1 / (1 + SM_output_ins.z_star()) );
    double ators_value_exp = 148.89;
    EXPECT_NEAR(ators_value_exp, ators_value, abs_err);
    
    double atolA_value = SM_output_ins.atolA(1 / (1 + SM_output_ins.z_star()) );
    double atolA_value_exp = 287.278;
    EXPECT_NEAR(atolA_value_exp, atolA_value, abs_err);

    std::cout << SM_output_ins.chisq_WMAP7_CMB_BAO() << "\n";
    double chisq_WMAP7_CMB_BAO_value = SM_output_ins.chisq_WMAP7_CMB_BAO();
    double chisq_WMAP7_CMB_BAO_value_exp = 508.418;
    EXPECT_NEAR(chisq_WMAP7_CMB_BAO_value_exp, chisq_WMAP7_CMB_BAO_value, abs_err);
    
    std::size_t z_sn_size_value = SM_output_ins.get_z_sn_size();
    std::size_t z_sn_size_value_exp = 580;
    EXPECT_EQ(z_sn_size_value_exp, z_sn_size_value);    
}

// bad values
TEST(SigmaCDM_model, test_2)
{
    
    std::map<std::string, double> cosm_params =
	{{"Omega_DEi", -5.553}, 
	 {"Omega_mi", -2.22098}};

    cosm_params["mu"] = 1.0;
    cosm_params["lam"] = 0.1;
    cosm_params["omegaDE_i"] = 0.0;

    SM_output SM_output_ins(cosm_params);
    SM_output_ins.solve_background();
    
    const double hub = 0.7;
    const double Omega_b0_default = 0.022765 * std::pow(hub, -2.0);
    
    SM_output_ins.setOmega_b0(Omega_b0_default);

    SM_output_ins.setOmega_curv0();

    SM_output_ins.setSN_data();
    SM_output_ins.chisq_SN_init();
    SM_output_ins.precompute_atochi_spline_table();
    SM_output_ins.chisq_BAO_CMB_init();
    SM_output_ins.precompute_ators_spline_table();
    SM_output_ins.precompute_atoR_spline_table();

    std::cout << SM_output_ins.chisq_SN() << "\n";
    double chisq_SN_value = SM_output_ins.chisq_SN();
    double chisq_SN_value_exp = 1.0e30;
    ASSERT_EQ(chisq_SN_value_exp, chisq_SN_value);

    std::cout << SM_output_ins.chisq_WMAP7_CMB_BAO() << "\n";
    double chisq_WMAP7_CMB_BAO_value = SM_output_ins.chisq_WMAP7_CMB_BAO();
    double chisq_WMAP7_CMB_BAO_value_exp = 1.0e30;
    ASSERT_EQ(chisq_WMAP7_CMB_BAO_value_exp, chisq_WMAP7_CMB_BAO_value);

}
