#ifndef _MCMC_HPP_
#define _MCMC_HPP_

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <armadillo>
#include <fstream>
#include <map>
#include <string>
#include <omp.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "utils.hpp"

struct MCMC{
    std::map<std::string, double> lower_bound;
    std::map<std::string, double> upper_bound;
    std::map<std::string, double> dpar;
    unsigned long NUMBER_OF_STEPS;
    double mh_factor;
    std::map<std::string, double> state;
    std::map<std::string, std::vector<double>> chains;
    std::vector<double> chisq_values;
    std::map<std::string, double> trial_state;

    MCMC(const std::map<std::string, double>& _lower_bound, const std::map<std::string, double>& _upper_bound, 
	 const std::map<std::string, double>& _dpar, 
	 unsigned long _NUMBER_OF_STEPS = 10, double _mh_factor = 1.0):
	lower_bound(_lower_bound), upper_bound(_upper_bound), dpar(_dpar), 
	NUMBER_OF_STEPS(_NUMBER_OF_STEPS), mh_factor(_mh_factor) {};

    double check_params(double lower_bound_value, double upper_bound_value, double new_step)
    {

	double checked_new_step = 0.0;

	if ( new_step < lower_bound_value) {
	    checked_new_step = lower_bound_value;
	}
	else if (new_step > upper_bound_value) 
	{
	    checked_new_step = upper_bound_value;
	}
	else checked_new_step = new_step;

	return checked_new_step;
    }

    double do_random_jump(double state_value, double some_random_jump, double dpar_value)
    {
	double trial_state_value = state_value + some_random_jump * dpar_value;
	return trial_state_value;

    }

    void print_state()
    {
	std::cout << "state printing" << "\n";
	for(const auto &param: state) 
	{
	    std::cout << param.first << " " << param.second << std::endl;
	}
    }

    void print_trial_state()
    {
	std::cout << "trial_state printing" << "\n";
	for(const auto &param: trial_state) 
	{
	    std::cout << param.first << " " << param.second << std::endl;
	}
    }
	
    void reset_chain()
    {
	std::random_device rd;
	std::mt19937 gen(rd());

	for(auto &param: chains)
	{
	    std::uniform_real_distribution<> dis_for_first_step(lower_bound.at(param.first), 
								upper_bound.at(param.first));
	    double step = dis_for_first_step(gen);
	    state.at(param.first) = step;
	    param.second.push_back(step);
	}
    }

    void start_chain()
    {
	for(const auto &param: lower_bound)
	    {
	    	std::vector<double> chain;
	    	chains.insert(std::make_pair(param.first, chain));
		double param_value = 0.0;
		trial_state.insert(std::make_pair(param.first, param_value));
		state.insert(std::make_pair(param.first, param_value));
	    }
	
	std::random_device rd;
	std::mt19937 gen(rd());

	for(auto &param: chains)
	{
	    std::uniform_real_distribution<> dis_for_first_step(lower_bound.at(param.first), 
								upper_bound.at(param.first));
	    double first_step = dis_for_first_step(gen);
	    state.at(param.first) = first_step; // always accept the first step
	    param.second.push_back(first_step);
	}
    }

    template <typename Func>
    void do_mcmc(Func& Fo){

	std::random_device rd;
	std::mt19937 gen(rd());
    
	std::normal_distribution<double> distn(0.0, 1.0);
	    
	for(const auto &param: lower_bound)
	{
	    std::vector<double> chain = {};
	    chains.insert(std::make_pair(param.first, chain));
	}

	start_chain();
	// print_chains();

	chisq_values.push_back(Fo.chisq(state));
    
	// main loop

	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini("../config.ini", pt);
	int RESET_NUMBER = pt.get<int>("Section1.RESET_NUMBER");
	    
	for (unsigned long i = 1; i < NUMBER_OF_STEPS; i++) {

	    if (i % (NUMBER_OF_STEPS / RESET_NUMBER) == 0){
		// std::cout << "reset i = " << i << "\n";
		reset_chain();
		chisq_values.push_back(Fo.chisq(state));
	    }

	    for(const auto &param: chains)
	    {
		double some_random_jump = distn(gen);
		double random_jump = do_random_jump(state.at(param.first), some_random_jump, dpar.at(param.first));
		double checked_trial_state = check_params(lower_bound.at(param.first), 
							  upper_bound.at(param.first), 
							  random_jump);
		trial_state.at(param.first) = checked_trial_state;
	    }

	    double chisq_old = chisq_values.back();
	    double chisq_new = Fo.chisq(trial_state);

	    if (std::isnan(chisq_new))
	    {
		chisq_new = 1.0e30;
	    }

	    std::uniform_real_distribution<> dist_a(0, 1);
	    double accept = dist_a(gen);
	    double possibly_large_number = -0.5 * (mh_factor * (chisq_new - chisq_old));
	    double atest = 0.0;
	    if (possibly_large_number > 300)
	    {
		atest = 1.1;
	    }
	    else 
	    {
		atest = exp(possibly_large_number);
	    }

	    if (std::min(1.0, atest) >= accept)
	    {
		for(auto &param: chains)
		{
		    chains.at(param.first).push_back(trial_state.at(param.first));
		    state.at(param.first) = trial_state.at(param.first);
		}
		chisq_values.push_back(chisq_new);
		// chisq_old = chisq_new;
	    }
	    else
	    {
		// for(auto &param: chains)
		// {
		//     chains.at(param.first).push_back(state.at(param.first));
		// }
		// chisq_values.push_back(chisq_old);
		continue;
	    }
	}
    }
    

    void print_chains(){

    	for(const auto &mcmc_param: chains)
    	{
    	    std::cout << mcmc_param.first << "\n";
	    std::cout << arma::vec(mcmc_param.second) << "\n";
    	}

	std::cout << arma::vec(chisq_values) << "\n";
	
    }

    void write_chains_to_hdf5(){

	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini("../config.ini", pt);
	std::string chain_file_name_prefix = pt.get<std::string>("Section1.chain_file_name_prefix");
	std::string chisq_chain_file_name_prefix = pt.get<std::string>("Section1.chisq_chain_file_name_prefix");

	std::string chain_file_name = chain_file_name_prefix + "0.h5";
	std::string chisq_chain_file_name = chisq_chain_file_name_prefix + "0.h5";
	
	int error_code;
	
	for(const auto &mcmc_param: chains)
    	{
	    std::string group_name = mcmc_param.first;
    	    error_code = write_array(chain_file_name, group_name, mcmc_param.second);

    	}

	error_code = write_array(chisq_chain_file_name, "chisq_values", chisq_values);

    }

};

#endif /* _MCMC_HPP_ */
