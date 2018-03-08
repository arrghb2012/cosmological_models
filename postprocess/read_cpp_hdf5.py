#-*- coding: utf-8 -*-
import numpy as np
import h5py
from itertools import combinations
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read('../config.ini')

NCHAINS = int(config.get('Section1', 'NCHAINS'))
NSTEPS = int(config.get('Section1', 'NSTEPS'))

##################################################
# получение списка имен параметров, имеющихся в цепи
##################################################

fname = config.get('Section1', 'chain_file_name_prefix') + "0" + ".h5"
g = h5py.File(fname, mode = 'r')
list_of_names = []
g.visit(list_of_names.append)

NPARAMS = len(list_of_names)

params_dict = {}

for i in range(NPARAMS):
    param_name = list_of_names[i]
    params_dict[param_name] = np.empty([0])

base_filename = config.get('Section1', 'chain_file_name_prefix')

for i in range(NCHAINS):
    filename = base_filename + str(i) + ".h5"
    # print(filename)
    g = h5py.File(filename, mode = 'r')
    for param in params_dict:
        # print(g[param].value)
        params_dict[param] = np.concatenate((params_dict[param], g[param].value))

##################################################
# извлечение значений chisq
##################################################

chisq_dict = {}

base_chisq_filename = config.get('Section1', 'chisq_chain_file_name_prefix')
fname = base_chisq_filename + "0" + ".h5"
g_chisq = h5py.File(fname, mode = 'r')
chisq_list_of_names = []
g_chisq.visit(chisq_list_of_names.append)

for i in range(len(chisq_list_of_names)):
    param_name = chisq_list_of_names[i]
    chisq_dict[param_name] = np.empty([0])

for i in range(NCHAINS):
    filename = base_chisq_filename + str(i) + ".h5"
    # print(filename)
    g = h5py.File(filename, mode = 'r')
    for param in chisq_dict:
        chisq_dict[param] = np.concatenate((chisq_dict[param], g[param].value))

# print(chisq_dict)

##################################################
# processing цепей - удаление дупликатов и
# значений, для которых chisq < threshold
##################################################

chisq_threshold = 700.0

for chisq in chisq_dict:
    # print(chisq)
    print(chisq_dict[chisq].shape) # почему 40008 а не 40000?
    a = chisq_dict[chisq].copy()
    print(a)
    print(np.where(a < chisq_threshold))
    print(a[np.where(a < chisq_threshold)].shape)
    new_params = np.where(a < chisq_threshold)
    # print((np.unique(chisq_dict[chisq])))
    for param in params_dict:
        print(param)
        print(params_dict[param].shape)
        # print(params_dict[param][new_params].shape)
        params_dict[param] = params_dict[param][new_params]
        # print((np.unique(params_dict[param])).shape)

for param in params_dict:
    print(param)
    print(params_dict[param].shape)

##################################################
# задание словаря для латех значений параметров вручную
##################################################

params_dict_latex_ini = dict(Omega_DEi = '$\Omega^{(i)}_{DE}$', \
    Omega_mi = '$\Omega^{(i)}_{m}$', mu = "$\mu$", \
    lam = "$\lambda$", omegaDE_i = "$\omega^{(i)}_{DE}$")

##################################################
# contours plotting
##################################################

import numpy as np
import matplotlib.pyplot as plt

##################################################
# общий подход
##################################################

from pdf_utils import get_1d_pdf
from pdf_utils import get_2d_posterior


def plot_1ddists(cosm_params_dict, latex_params_dict):
    for param_name, params_value in cosm_params_dict.iteritems():
        params_lb_value = np.float64(config.get('Section_lower_bounds', param_name))
        params_ub_value = np.float64(config.get('Section_upper_bounds', param_name))
        latex_param_name = latex_params_dict[param_name]

        pdf_ins = get_1d_pdf(params_value, params_lb_value, params_ub_value)
        pdf_ins.set_sigma_level()
        pdf_ins.prepare_histogram_from_subdivision(cut_off = 0, bins_number = 40)
        pdf_ins.prepare_get_interpolated_1d_distr()

        plt.plot(pdf_ins.x, pdf_ins.get_interpolated_1d_distr(pdf_ins.x), '-', color = 'black', linewidth = 3)
        plt.xlabel(latex_param_name, {'fontsize': 20})

        plt.ylim(0.0, 1.2)
        plt.xlim(params_lb_value, params_ub_value)

        plt.tight_layout(pad=0.1)  # Make the figure use all available whitespace

        figname = param_name + "_1d" + ".pdf"
        plt.savefig(figname)
        plt.cla()


def plot_1ddists_1(cosm_params_dict, latex_params_dict):
    for param_name, params_value in cosm_params_dict.iteritems():
        params_lb_value = np.float64(config.get('Section_lower_bounds', param_name))
        params_ub_value = np.float64(config.get('Section_upper_bounds', param_name))
        latex_param_name = latex_params_dict[param_name]

        pdf_ins = get_1d_pdf(params_value, params_lb_value, params_ub_value)
        pdf_ins.set_sigma_level()
        pdf_ins.prepare_histogram_from_subdivision_ext(cut_off = 0, bins_number = 40)
        pdf_ins.prepare_get_interpolated_1d_distr()

        plt.plot(pdf_ins.x, pdf_ins.get_interpolated_1d_distr(pdf_ins.x), '-', color = 'black', linewidth = 3)
        plt.xlabel(latex_param_name, {'fontsize': 20})

        plt.ylim(0.0, 1.2)
        plt.xlim(params_lb_value, params_ub_value)

        plt.tight_layout(pad=0.1)  # Make the figure use all available whitespace

        figname = param_name + "_1d" + ".pdf"
        plt.savefig(figname)
        plt.cla()


print("=============================")
plot_1ddists(params_dict, params_dict_latex_ini)    
print("=============================")

params_dict_plot2d_lims = dict(Omega_DEi = [-25.0, -1.34678748622], \
                               Omega_mi = [-10.0, 0.0], \
                               mu = [0.01, 10.0], \
                               lam = [0.01, 10.0], \
                               omegaDE_i = [-1.0, 1.0])

def plot_2dcontours(cosm_params_dict, latex_params_dict):

    result_list = list(map(dict, combinations(
        cosm_params_dict.items(), 2)))

    for params_pair in result_list:

        first_param_name, first_param_values = params_pair.popitem()
        second_param_name, second_param_values = params_pair.popitem()

        contour_ins = get_2d_posterior(first_param_values, second_param_values)
        contour_ins.prepare_contour()

        fig = plt.figure()
        plt.plot(first_param_values, second_param_values, 'o', ms = 4, alpha = 0.1, color = 'grey')

        plt.xlim(params_dict_plot2d_lims[first_param_name][0], \
                 params_dict_plot2d_lims[first_param_name][1])

        plt.ylim(params_dict_plot2d_lims[second_param_name][0], \
                 params_dict_plot2d_lims[second_param_name][1])

        plt.xlabel(params_dict_latex_ini[first_param_name], {'fontsize': 20})
        plt.ylabel(params_dict_latex_ini[second_param_name], {'fontsize': 20})

        plt.tight_layout(pad=0.1)  # Make the figure use all available whitespace
        figname = first_param_name + "_vs_" + second_param_name + ".pdf"
        plt.savefig(figname)
        plt.cla()


print("=============================")
plot_2dcontours(params_dict, params_dict_latex_ini)
print("=============================")
