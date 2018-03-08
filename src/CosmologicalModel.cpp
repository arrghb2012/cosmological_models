#include "CosmologicalModel.hpp"

const double pi = 4.0 * std::atan(1.0);

double CosmologicalModel::atoH(double a){

    double atoH_value = getOmega_m0() * std::pow(a, -3) + Omega_r0 * std::pow(a, -4) +
    				  getOmega_DE0() + getOmega_curv0() * std::pow(a, -2);

    if (atoH_value < 0)
    	throw std::invalid_argument("get complex hubble parameter");

    return std::sqrt(atoH_value);
}

void CosmologicalModel::setOmega_b0(double Omega_b0) {
    mOmega_b0 = Omega_b0;
}

double CosmologicalModel::getOmega_b0() const{
    return mOmega_b0;
}

void CosmologicalModel::setOmega_m0(double Omega_m0) {
    mOmega_m0 = Omega_m0;
}

double CosmologicalModel::getOmega_m0() const{
    return mOmega_m0;
}

void CosmologicalModel::setOmega_DE0(double Omega_DE0){
    mOmega_DE0 = Omega_DE0;
}

double CosmologicalModel::getOmega_DE0() const{
    return mOmega_DE0;
}

void CosmologicalModel::setOmega_curv0(){
    mOmega_curv0 = 1.0 - getOmega_m0() - getOmega_DE0() - Omega_r0;
}

double CosmologicalModel::getOmega_curv0() const{
    return mOmega_curv0;
}

void CosmologicalModel::setSN_data(){

    double z_sn_temp, mu_exp_temp, mu_err_temp;
    std::string input_file_name = "../data/SCPUnion2.1_mu_vs_z_modified.txt";
    std::ifstream my_in(input_file_name);
    if (my_in.fail()) {
	std::cerr << "Unable to open the SN data file " << input_file_name
		  << " for input." << std::endl;
	std::exit(EXIT_FAILURE);
    }

    while (my_in >> z_sn_temp >> mu_exp_temp >> mu_err_temp) {
    	z_sn.push_back(z_sn_temp);
    	mu_exp.push_back(mu_exp_temp);
    	mu_err.push_back(mu_err_temp);
    }

}

std::size_t CosmologicalModel::get_z_sn_size(){
    return z_sn.size();
}

void CosmologicalModel::chisq_SN_init(){
    a_array_for_chisq_SN = arma::exp10(arma::linspace<arma::vec>(-4.0, 0.0, 1000));
    a_array_size_for_chisq_SN = a_array_for_chisq_SN.size();
    atodchi_array = arma::zeros<arma::vec>(a_array_size_for_chisq_SN);
    std::string type = "cubic";
    cubic_spline_for_atodchi.alloc_Spline(a_array_size_for_chisq_SN, type);
}

void CosmologicalModel::precompute_atochi_spline_table()
{
    for (unsigned int i = 0; i < a_array_size_for_chisq_SN; ++i)
    	{
    	    try {
    		atodchi_array(i) = 1.0 / (std::pow(a_array_for_chisq_SN(i), 2.0) * this->atoH(a_array_for_chisq_SN(i)));
    	    }
    	    catch (const std::invalid_argument& e) {
    		 std::cout << "Caught exception: " << e.what() << std::endl;
    		cubic_spline_for_atodchi.setup_Spline(&a_array_for_chisq_SN[0], &atodchi_array[0], a_array_size_for_chisq_SN);
    		chisq_SN_value = 1.0e30;
    		chisq_WMAP7_CMB_BAO_value = 1.0e30;
    		chisq_Planck_value = 1.0e30;
    		chisq_WMAP9_value = 1.0e30;
		chisq_GC_value = 1.0e30;
    		return;
    	    }
    	}

    cubic_spline_for_atodchi.setup_Spline(&a_array_for_chisq_SN[0], &atodchi_array[0], a_array_size_for_chisq_SN);

}

double CosmologicalModel::atodL(double a)
{
    double atochi_value, d_L;

    atochi_value = cubic_spline_for_atodchi.y_integrate(a, 1.0);

    if (getOmega_curv0() > 0){
	d_L = 1 / H_0 / (a * std::sqrt(std::abs(getOmega_curv0()))) * 
	    std::sinh(std::sqrt(std::abs(getOmega_curv0())) * atochi_value);
    }
    else if (getOmega_curv0() < 0) {
	d_L = 1 / H_0 / (a * std::sqrt(std::abs(getOmega_curv0()))) * 
	    std::sin(std::sqrt(std::abs(getOmega_curv0())) * atochi_value);
    }
    else  d_L = 1 / H_0 / a * atochi_value;

    return d_L;
}

double CosmologicalModel::atomu(double a){
    return 5.0 * std::log10(this->atodL(a)) + 25;
}

double CosmologicalModel::chisq_SN()
{

    if (chisq_SN_value == 1.0e30) {
	return chisq_SN_value;
    }
	
    double aA = 0.0, aB = 0.0, Csn = 0.0, cchi_SN = 0.0;

    double atomu_temp = 0.0, denom_temp = 0.0;

    for (unsigned int i = 0; i < z_sn.size(); ++i)
    	{
    	    denom_temp = mu_err[i] * mu_err[i];
    	    atomu_temp = mu_exp[i] - this->atomu(1/(1+z_sn[i]));
    	    Csn += 1 / denom_temp;
    	    aA += atomu_temp * atomu_temp / denom_temp;
    	    aB += atomu_temp / denom_temp;
    	}

    cchi_SN = aA - aB * aB / Csn;

    return cchi_SN;

}

void CosmologicalModel::chisq_BAO_CMB_init(){
    a_array_atodrs = arma::exp10(arma::linspace<arma::vec>(-6.0, 0.0, 1000));
    a_array_atodR = arma::exp10(arma::linspace<arma::vec>(-4.0, 0.0, 1000));
    a_array_atodrs_size = a_array_atodrs.size();
    a_array_atodR_size = a_array_atodR.size();
    atodrs_array = arma::zeros<arma::vec>(a_array_atodrs_size);
    atodR_array = arma::zeros<arma::vec>(a_array_atodR_size);
    std::string type = "cubic";
    cubic_spline_atodrs.alloc_Spline(a_array_atodrs_size, type);
    cubic_spline_for_atoR.alloc_Spline(a_array_atodR_size, type);

}

double CosmologicalModel::atoDV(double a){
    return std::pow(std::pow(a, 2.0) * std::pow(this->atodL(a), 2.0) * (1.0 / a - 1.0) / atoH(a), 1.0 / 3);
}

double CosmologicalModel::atodrs(double a){
    return 1.0 / H_0 / (std::pow(a, 2.0) * atoH(a) *
			std::sqrt(1 + (3 * getOmega_b0()/ 4.0 / Omega_r0) * a)) / std::sqrt(3.0);
}

void CosmologicalModel::precompute_ators_spline_table()
{

    for (unsigned int i = 0; i < a_array_atodrs_size; ++i)
    	{
    	    try {
    		atodrs_array(i) = atodrs(a_array_atodrs(i));
    	    }
    	    catch (const std::invalid_argument& e) {
    		cubic_spline_atodrs.setup_Spline(&a_array_atodrs[0], &atodrs_array[0], a_array_atodrs_size);
    		chisq_WMAP7_CMB_BAO_value = 1.0e30;
		    chisq_GC_value = 1.0e30;
    		return;
    	    }
    	}

    cubic_spline_atodrs.setup_Spline(&a_array_atodrs[0], &atodrs_array[0], a_array_atodrs_size);

}

double CosmologicalModel::ators(double a){

    double ators_value = 0.0;
    ators_value = cubic_spline_atodrs.y_integrate(a_array_atodrs[0], a);

    return ators_value;
}

void CosmologicalModel::precompute_atoR_spline_table()
{

    for (unsigned int i = 0; i < a_array_atodR_size; ++i)
    	{
    	    try {
    		atodR_array(i) = 1.0 / (std::pow(a_array_atodR(i), 2.0) * atoH(a_array_atodR(i)));
    	    }
    	    catch (const std::invalid_argument& e) {
    		cubic_spline_for_atoR.setup_Spline(&a_array_atodR[0], &atodR_array[0], a_array_atodR_size);
    		chisq_SN_value = 1.0e30;
    		chisq_WMAP7_CMB_BAO_value = 1.0e30;
    		chisq_Planck_value = 1.0e30;
    		chisq_WMAP9_value = 1.0e30;
    		return;
    	    }
    	}

    cubic_spline_for_atoR.setup_Spline(&a_array_atodR[0], &atodR_array[0], a_array_atodR_size);

}

double CosmologicalModel::atoR(double a)
{
    double ator_value = 0.0, R = 0.0;

    ator_value = cubic_spline_for_atoR.y_integrate(a, 1.0);

    if (getOmega_curv0() > 0){
	R = std::sqrt(getOmega_m0()) / (std::sqrt(std::abs(getOmega_curv0()))) *
	    std::sinh(std::sqrt(std::abs(getOmega_curv0())) * ator_value);
    }
    else if (getOmega_curv0() < 0) {
	R = std::sqrt(getOmega_m0()) / (std::sqrt(std::abs(getOmega_curv0()))) *
	    std::sinh(std::sqrt(std::abs(getOmega_curv0())) * ator_value);

    }
	
    else R = std::sqrt(getOmega_m0()) * ator_value;

    return R;

}
    
double CosmologicalModel::z_star(){
    double g1 = (0.0783 * std::pow((getOmega_b0() * std::pow(hub, 2)), -0.238)) /
	(1 + 39.5 * std::pow((getOmega_b0() * std::pow(hub, 2)), 0.763));
    double g2 = 0.560 / (1 + 21.1 * std::pow((getOmega_b0() * std::pow(hub, 2)), 1.81));
    double z_star_value = 1048 * (1 + 0.00124 * std::pow((getOmega_b0() * std::pow(hub, 2)), (-0.738))) *
	(1 + g1 * std::pow((getOmega_m0() * std::pow(hub, 2)), g2));
    return z_star_value;
}

double CosmologicalModel::z_drag(){
    double b1 = (0.313 * std::pow((getOmega_m0() * std::pow(hub, 2)), -0.419)) /
	(1 + 0.607 * std::pow((getOmega_m0() * std::pow(hub, 2)), 0.674));
    double b2 = 0.238 * std::pow((getOmega_m0() * std::pow(hub, 2)), 0.223);
    double z_drag_value = 1291 * (std::pow((getOmega_m0() * std::pow(hub, 2)), 0.251)) *
	(1 + b1 * std::pow((getOmega_b0() * std::pow(hub, 2)), b2)) / (1 + 0.659 * std::pow(getOmega_m0() * std::pow(hub, 2), 0.828));
    return z_drag_value;
}

double CosmologicalModel::atolA(double a){
    double atolA_value = pi * this->atodL(a) * a / this->ators(a);
    return atolA_value;
}

double CosmologicalModel::atoDA(double a){
    double atoDA_value = std::pow(a, 2) * atodL(a) * H_0;
    return atoDA_value;
}

double CosmologicalModel::chisq_WMAP7_CMB_BAO()
{    

    if (chisq_WMAP7_CMB_BAO_value == 1.0e30) {
	return chisq_WMAP7_CMB_BAO_value;
    }

    double z_star_value = z_star();
    arma::vec CMBfit_vector = {atolA(1 / ( z_star_value + 1 )), atoR(1 / ( z_star_value + 1 )), 
			       z_star_value};
     std::cout << CMBfit_vector;
    arma::mat CCMB = {2.305, 29.698, -1.333, 29.698, 6825.27, -113.180, -1.333, -113.180, 3.414};
    CCMB.reshape(3, 3);
    arma::vec CMBfit_vector_WMAP = {302.09, 1.725, 1091.3};
    double cchi_CMB = 0.0, cchi_BAO = 0.0;
     cchi_CMB = arma::dot(CMBfit_vector - CMBfit_vector_WMAP, arma::dot(CCMB, CMBfit_vector - CMBfit_vector_WMAP));
    arma::vec cchi_CMB_intm = arma::zeros<arma::vec>(3);
     arma::vec cchi_CMB_intm;
    for (unsigned int i = 0; i < CCMB.n_rows; ++i)
	{
	    cchi_CMB_intm(i) = arma::dot(CCMB.row(i), CMBfit_vector - CMBfit_vector_WMAP);
	}
    cchi_CMB = arma::dot(CMBfit_vector - CMBfit_vector_WMAP, cchi_CMB_intm);
     std::cout << "cchi_CMB " << cchi_CMB << "\n";
    cchi_BAO = std::pow((atoDV(1 / ( 1 + 0.35 )) /
			 atoDV(1 / ( 1 + 0.2 )) - 1.736 ), 2) / std::pow(0.065, 2);
     std::cout << "cchi_BAO " << cchi_BAO << "\n";
    chisq_WMAP7_CMB_BAO_value = cchi_CMB + cchi_BAO;
    return chisq_WMAP7_CMB_BAO_value;
}

double CosmologicalModel::chisq_Planck()
{    

    if (chisq_Planck_value == 1.0e30) {
	return chisq_Planck_value;
    }

    double z_star_value = z_star();
    arma::vec fit_vector = {atolA(1 / ( z_star_value + 1 )), atoR(1 / ( z_star_value + 1 )), 
    			       getOmega_b0() * std::pow(hub, 2)}; // (l_a,R,Omega_b)

    arma::mat Norm_Cov_mat_Planck = {1.0000, 0.5250, -0.4235, 0.5250, 1.0000, -0.6925, -0.4235, -0.6925, 1.0000};
    Norm_Cov_mat_Planck.reshape(3, 3);
    arma::vec sigma_vec_Planck = {0.18, 0.0094, 0.00030};

    arma::mat Cov_mat_Planck = arma::zeros<arma::mat>(3, 3);

    for(unsigned int i = 0; i < Cov_mat_Planck.n_rows; i++){
	for(unsigned int j = 0; j < Cov_mat_Planck.n_cols; j++){
            Cov_mat_Planck(i, j) = sigma_vec_Planck(i) * sigma_vec_Planck(j) * Norm_Cov_mat_Planck(i, j);
	} 
    }

    arma::mat Cov_mat_Planck_inv = Cov_mat_Planck.i();
    arma::vec fit_vector_Planck = {301.57, 1.7407, 0.02228};

    arma::vec cchi_Planck_intm = arma::zeros<arma::vec>(3);

    for (unsigned int i = 0; i < Cov_mat_Planck_inv.n_rows; ++i)
	{
	    cchi_Planck_intm(i) = arma::dot(Cov_mat_Planck_inv.row(i), fit_vector - fit_vector_Planck);
	}
    chisq_Planck_value = arma::dot(fit_vector - fit_vector_Planck, cchi_Planck_intm);

    return chisq_Planck_value;
}

double CosmologicalModel::chisq_WMAP9()
{    

    if (chisq_WMAP9_value == 1.0e30) {
	return chisq_WMAP9_value;
    }

    double z_star_value = z_star();
    arma::vec fit_vector = {atolA(1 / ( z_star_value + 1 )), atoR(1 / ( z_star_value + 1 )), 
    			       getOmega_b0() * std::pow(hub, 2)}; // (l_a,R,Omega_b)

    arma::mat Norm_Cov_mat_WMAP9 = {1.0000, 0.3883, -0.6089, 0.3883, 1.0000, -0.5239, -0.6089, -0.5239, 1.0000};
    Norm_Cov_mat_WMAP9.reshape(3, 3);
    arma::vec sigma_vec_WMAP9 = {0.66, 0.0164, 0.00053};

    arma::mat Cov_mat_WMAP9 = arma::zeros<arma::mat>(3, 3);

    for(unsigned int i = 0; i < Cov_mat_WMAP9.n_rows; i++){
	for(unsigned int j = 0; j < Cov_mat_WMAP9.n_cols; j++){
            Cov_mat_WMAP9(i, j) = sigma_vec_WMAP9(i) * sigma_vec_WMAP9(j) * Norm_Cov_mat_WMAP9(i, j);
	} 
    }

    arma::mat Cov_mat_WMAP9_inv = Cov_mat_WMAP9.i();
    arma::vec fit_vector_WMAP9 = {302.02, 1.7327, 0.02260};

    arma::vec cchi_WMAP9_intm = arma::zeros<arma::vec>(3);

    for (unsigned int i = 0; i < Cov_mat_WMAP9_inv.n_rows; ++i)
	{
	    cchi_WMAP9_intm(i) = arma::dot(Cov_mat_WMAP9_inv.row(i), fit_vector - fit_vector_WMAP9);
	}
    chisq_WMAP9_value = arma::dot(fit_vector - fit_vector_WMAP9, cchi_WMAP9_intm);

    return chisq_WMAP9_value;
}



double CosmologicalModel::chisq_GC()
{
    if (chisq_GC_value == 1.0e30) {
	return chisq_GC_value;
    }

    double z_GC_1 = 0.35;

    static double atoH_value = 0.0;

    try {
	atoH_value = atoH(1/(z_GC_1+1));
    }
    catch (const std::invalid_argument& e) {
	chisq_GC_value = 1.0e30;
	return chisq_GC_value;
    }

    double z_drag_value = z_drag();

    double chisq_GC_1_value = 0.0, chisq_GC_2_value = 0.0;

    arma::vec fit_vector_1 = {atoH_value * ators(1 / (z_drag_value + 1)), \
    			      atoDA(1 / (z_GC_1 + 1)) / ators(1 / (z_drag_value + 1))};
   
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

    try {
	atoH_value = atoH(1/(z_GC_2 + 1));
    }
    catch (const std::invalid_argument& e) {
	chisq_GC_value = 1.0e30;
	return chisq_GC_value;
    }

    arma::vec fit_vector_2 = {atoH_value * ators(1 / (z_drag_value + 1)), \
    			      atoDA(1 / (z_GC_2 + 1)) / ators(1 / (z_drag_value + 1))};

    arma::mat Norm_Cov_mat_GC_2 = {1.0000, 0.0604, 0.0604, 1.0000};
    Norm_Cov_mat_GC_2.reshape(2, 2);
    arma::vec sigma_vec_GC_2 = {0.0031, 0.27};

    arma::mat Cov_mat_GC_2 = arma::zeros<arma::mat>(2, 2);

    for(unsigned int i = 0; i < Cov_mat_GC_2.n_rows; i++){
	for(unsigned int j = 0; j < Cov_mat_GC_2.n_cols; j++){
            Cov_mat_GC_2(i, j) = sigma_vec_GC_2(i) * sigma_vec_GC_2(j) * Norm_Cov_mat_GC_2(i, j);
	} 
    }

    arma::mat Cov_mat_GC_2_inv = Cov_mat_GC_2.i();
    arma::vec fit_vector_GC_2 = {0.0484, 8.95};

    arma::vec cchi_GC_2_intm = arma::zeros<arma::vec>(2);

    for (unsigned int i = 0; i < Cov_mat_GC_2_inv.n_rows; ++i)
	{
	    cchi_GC_2_intm(i) = arma::dot(Cov_mat_GC_2_inv.row(i), fit_vector_2 - fit_vector_GC_2);
	}
    chisq_GC_2_value = arma::dot(fit_vector_2 - fit_vector_GC_2, cchi_GC_2_intm);

    chisq_GC_value = chisq_GC_1_value + chisq_GC_2_value;
    
    return chisq_GC_value;
}



