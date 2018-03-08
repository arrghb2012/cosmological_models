#include <chrono>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "LambdaCDM.hpp"
#include "utils.hpp"
#include "mcmc.hpp"

class chisq_joint_class
{
public:
    chisq_joint_class() {};
    ~chisq_joint_class() {};

    void chisq_joint_class_init() {
	LCDM_model_ins.setSN_data();
	LCDM_model_ins.chisq_SN_init();
	LCDM_model_ins.chisq_BAO_CMB_init();
    }

    double chisq(const std::map<std::string, double>& state) {

	const double hub = 0.7;
	const double Omega_b0_default = 0.022765 * std::pow(hub, -2.0);
	LCDM_model_ins.setOmega_m0(state.at("Omega_m0"));
	LCDM_model_ins.setOmega_DE0(state.at("Omega_DE0"));
	LCDM_model_ins.setOmega_b0(Omega_b0_default);
	LCDM_model_ins.setOmega_curv0();

	LCDM_model_ins.precompute_atochi_spline_table();
	LCDM_model_ins.precompute_ators_spline_table();
	LCDM_model_ins.precompute_atoR_spline_table();

	 std::cout << LCDM_model_ins.atoH(0.5) << "\n";
	 std::cout << LCDM_model_ins.get_z_sn_size() << "\n";
	 std::cout << LCDM_model_ins.atodL(0.5) << "\n";
	 std::cout << LCDM_model_ins.atomu(0.5) << "\n";
	 std::cout << LCDM_model_ins.chisq_SN() << "\n";
	 std::cout << LCDM_model_ins.chisq_WMAP7_CMB_BAO() << "\n";

	 std::cout << LCDM_model_ins.atodL(0.5) << "\n";
	 std::cout << LCDM_model_ins.getOmega_curv0() << "\n";

	 return LCDM_model_ins.chisq_SN() + LCDM_model_ins.chisq_WMAP7_CMB_BAO();
    }

private:
    LambdaCDM_model LCDM_model_ins;

};

class chisq_joint_Planck_class
{
public:
    chisq_joint_Planck_class() {};
    ~chisq_joint_Planck_class() {};

    void chisq_joint_Planck_class_init() {
	LCDM_model_ins.setSN_data();
	LCDM_model_ins.chisq_SN_init();
	LCDM_model_ins.chisq_BAO_CMB_init();
    }

    double chisq(const std::map<std::string, double>& state) {

	LCDM_model_ins.setOmega_m0(state.at("Omega_m0"));
	LCDM_model_ins.setOmega_DE0(state.at("Omega_DE0"));
	LCDM_model_ins.setOmega_b0(state.at("Omega_b0"));
	LCDM_model_ins.setOmega_curv0();

	LCDM_model_ins.precompute_atochi_spline_table();
	LCDM_model_ins.precompute_ators_spline_table();
	LCDM_model_ins.precompute_atoR_spline_table();

	 std::cout << LCDM_model_ins.atoH(0.5) << "\n";
	 std::cout << LCDM_model_ins.get_z_sn_size() << "\n";
	 std::cout << LCDM_model_ins.atodL(0.5) << "\n";
	 std::cout << LCDM_model_ins.atomu(0.5) << "\n";
	 std::cout << LCDM_model_ins.chisq_SN() << "\n";
	 std::cout << LCDM_model_ins.chisq_WMAP7_CMB_BAO() << "\n";
	
	 std::cout << "LCDM_model_ins.atodL(0.5)" << "\n";
	 std::cout << LCDM_model_ins.atodL(0.5) << "\n";
	 std::cout << "LCDM_model_ins.chisq_Planck()" << "\n";
	 std::cout << LCDM_model_ins.chisq_Planck() << "\n";
	 std::cout << "LCDM_model_ins.chisq_WMAP9()" << "\n";
	 std::cout << LCDM_model_ins.chisq_WMAP9() << "\n";
	 std::cout << "LCDM_model_ins.chisq_GC()" << "\n";
	 std::cout << LCDM_model_ins.chisq_GC() << "\n";

	return LCDM_model_ins.chisq_SN() + LCDM_model_ins.chisq_Planck();
    }

private:
    LambdaCDM_model LCDM_model_ins;

};

class chisq_joint_WMAP9_class
{
public:
    chisq_joint_WMAP9_class() {};
    ~chisq_joint_WMAP9_class() {};

    void chisq_joint_WMAP9_class_init() {
	LCDM_model_ins.setSN_data();
	LCDM_model_ins.chisq_SN_init();
	LCDM_model_ins.chisq_BAO_CMB_init();
    }

    double chisq(const std::map<std::string, double>& state) {

	LCDM_model_ins.setOmega_m0(state.at("Omega_m0"));
	LCDM_model_ins.setOmega_DE0(state.at("Omega_DE0"));
	LCDM_model_ins.setOmega_b0(state.at("Omega_b0"));
	LCDM_model_ins.setOmega_curv0();

	LCDM_model_ins.precompute_atochi_spline_table();
	LCDM_model_ins.precompute_ators_spline_table();
	LCDM_model_ins.precompute_atoR_spline_table();

	return LCDM_model_ins.chisq_SN() + LCDM_model_ins.chisq_WMAP9();
    }

private:
    LambdaCDM_model LCDM_model_ins;

};

int main()
{

    {
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini("../config.ini", pt);
	unsigned long NUMBER_OF_STEPS = pt.get<unsigned long>("Section1.NSTEPS");

	std::map<std::string, double> lower_bound =
	    {{"Omega_DE0", pt.get<double>("Section_lower_bounds.Omega_DE0")},
	     {"Omega_m0", pt.get<double>("Section_lower_bounds.Omega_m0")}};

	std::map<std::string, double> upper_bound;
	std::map<std::string, double> dpar;

	for (const auto & param: lower_bound){
	    upper_bound[param.first] = pt.get<double>("Section_upper_bounds." + param.first);
	    dpar[param.first] = pt.get<double>("Section_dpar." + param.first);
	}

	auto start_s = std::chrono::system_clock::now();
    
	MCMC MCMC_ins{lower_bound, upper_bound, dpar, .NUMBER_OF_STEPS = NUMBER_OF_STEPS};
	chisq_joint_class chisq_joint_class_ins;
	chisq_joint_class_ins.chisq_joint_class_init();
	MCMC_ins.do_mcmc(chisq_joint_class_ins);
	MCMC_ins.write_chains_to_hdf5();

	auto end_s = std::chrono::system_clock::now();
	auto diff_s = end_s - start_s;
	std::cout << std::chrono::duration < double > (diff_s).count() << "s" << std::endl;

    }

}


