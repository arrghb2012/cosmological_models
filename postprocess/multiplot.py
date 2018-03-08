# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations_with_replacement, product
import h5py
import ConfigParser
import subprocess

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
    g = h5py.File(filename, mode = 'r')
    for param in params_dict:
        params_dict[param] = np.concatenate((params_dict[param], g[param].value))

##################################################
# извлечение значений chisq
##################################################

chisq_dict = {}

base_chisq_filename = config.get('Section1', 'chisq_chain_file_name_prefix')
print(base_chisq_filename)
fname = base_chisq_filename + "0" + ".h5"
g_chisq = h5py.File(fname, mode = 'r')
chisq_list_of_names = []
g_chisq.visit(chisq_list_of_names.append)

for i in range(len(chisq_list_of_names)):
    param_name = chisq_list_of_names[i]
    chisq_dict[param_name] = np.empty([0])

for i in range(NCHAINS):
    filename = base_chisq_filename + str(i) + ".h5"
    g = h5py.File(filename, mode = 'r')
    for param in chisq_dict:
        chisq_dict[param] = np.concatenate((chisq_dict[param], g[param].value))

##################################################
# processing цепей - удаление дупликатов и
# значений, для которых chisq < threshold
##################################################

chisq_threshold = 700.0

for chisq in chisq_dict:
    a = chisq_dict[chisq].copy()
    new_params = np.where(a < chisq_threshold)
    for param in params_dict:
        params_dict[param] = params_dict[param][new_params]

##################################################
# задание словаря для латех значений параметров вручную
##################################################

params_latex_dict = {}

for param in params_dict:
    params_latex_dict[param] = config.get('Section2', param)

print(params_latex_dict)

##################################################
# общий подход
##################################################

import matplotlib.pyplot as plt
from itertools import combinations

from pdf_utils import get_1d_pdf
from pdf_utils import get_2d_posterior

params_lims_dict = {}

for param in params_dict:
    params_lims_dict[param] = [np.float64(config.get('Section_lower_bounds', param)), \
                                      np.float64(config.get('Section_upper_bounds', param))]

print(params_lims_dict)

mean_values_dict = {}           # значения, которые будут
# использованы для построения параметра Хаббла и остальных величин

for param in params_dict:
    mean_values_dict[param] = 0.0

def plot_1ddists(cosm_params_dict, latex_params_dict):
    for param_name, params_value in cosm_params_dict.iteritems():

        params_lb_value = params_lims_dict[param_name][0]
        params_ub_value = params_lims_dict[param_name][1]

        latex_param_name = latex_params_dict[param_name]
        
        pdf_ins = get_1d_pdf(params_value, params_lb_value, params_ub_value)
        pdf_ins.set_sigma_level()
        pdf_ins.prepare_histogram_from_subdivision(cut_off = 0, bins_number = 100)
        pdf_ins.prepare_get_interpolated_1d_distr()
        
        plt.plot(pdf_ins.x, pdf_ins.get_interpolated_1d_distr(pdf_ins.x), '-', color = 'black', linewidth = 3)
        plt.xlabel(latex_param_name, {'fontsize': 20})
        plt.ylim(0.0, 1.1)
        plt.xlim(pdf_ins.x[0], pdf_ins.x[-1])
        mean_values_dict[param_name] = pdf_ins.x[np.argmax(pdf_ins.get_interpolated_1d_distr(pdf_ins.x))]

        params_lims_dict[param_name] = [pdf_ins.x[0], pdf_ins.x[-1]]

        plt.tight_layout(pad=0.1)  # Make the figure use all available whitespace
        figname = param_name + "_1d" + ".pdf"
        plt.savefig(figname)
        plt.cla()

def plot_1ddists_1(cosm_params_dict, latex_params_dict):
    for param_name, params_value in cosm_params_dict.iteritems():

        params_lb_value = params_lims_dict[param_name][0]
        params_ub_value = params_lims_dict[param_name][1]
        latex_param_name = latex_params_dict[param_name]
        
        pdf_ins = get_1d_pdf(params_value, params_lb_value, params_ub_value)
        pdf_ins.set_sigma_level()
        pdf_ins.prepare_histogram_from_subdivision_ext(cut_off = 0, bins_number = 20)
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
plot_1ddists(params_dict, params_latex_dict)    
print("mean_values_dict")
print(mean_values_dict)
print("=============================")


def plot_2dcontours(cosm_params_dict, latex_params_dict):

    result_list = list(map(dict, combinations(
        cosm_params_dict.items(), 2)))
    
    for params_pair in result_list:

        first_param_name, first_param_values = params_pair.popitem()
        second_param_name, second_param_values = params_pair.popitem()

        contour_ins = get_2d_posterior(first_param_values, second_param_values)
        contour_ins.prepare_contour()

        fig = plt.figure()
        plt.plot(first_param_values[::10], second_param_values[::10], 'o', ms = 4, alpha = 0.1, color = 'grey')
        plt.plot(contour_ins.x_vert_1_s, contour_ins.y_vert_1_s, "-", linewidth = 3, color = 'black')
        plt.plot(contour_ins.x_vert_2_s, contour_ins.y_vert_2_s, "-", linewidth = 3, color = 'black')

        plt.xlim(params_lims_dict[first_param_name][0], \
                 params_lims_dict[first_param_name][1])

        plt.ylim(params_lims_dict[second_param_name][0], \
                 params_lims_dict[second_param_name][1])

        plt.xlabel(params_latex_dict[first_param_name], {'fontsize': 20})
        plt.ylabel(params_latex_dict[second_param_name], {'fontsize': 20})

        plt.tight_layout(pad=0.1)  # Make the figure use all available whitespace
        figname = first_param_name + "_vs_" + second_param_name + ".pdf"
        plt.savefig(figname)
        plt.cla()


print("=============================")
plot_2dcontours(params_dict, params_latex_dict)
print("=============================")

#####################################################
# gnuplot 5.0 multiplot
#####################################################

class multiplot(object):
    def __init__(self, gnuplot_filename):
        self.gnuplot_filename = gnuplot_filename
        self.file_desc = open(str(self.gnuplot_filename) + ".plt", "w")

    def set_plot(self, layout):
        self.layout = layout
        self.file_desc.write("set terminal epslatex size 8, 8 standalone color colortext 10\n")
        output = "set output \'" + str(self.gnuplot_filename) + ".tex\'\n"
        self.file_desc.write(output)
        lt = "set multiplot layout " + str(layout) + "," + str(layout) + " \\\n"
        self.file_desc.write(lt)
        self.file_desc.write("              margins 0.1,0.98,0.1,0.98 \\\n")
        self.file_desc.write("              spacing 0.08,0.08\n")
        self.file_desc.write("set style line  1 lc rgb '#000000' pt 1 ps 0.2 lt 1 lw 3    # black\n")
        self.file_desc.write("set style line  2 lc rgb '#77ac30' pt 5 lw 1     # blue\n")

    def set_data_to_plot(self, params_dict, params_lims_dict, params_latex_dict):
        list_of_names = params_dict.keys()
        print(list_of_names)

        product_list = map(list, product(list_of_names, repeat = 2))

        for pair in product_list:
            row = list_of_names.index(pair[0])
            column = list_of_names.index(pair[1])

            self.file_desc.write("unset xlabel\n")
            self.file_desc.write("unset ylabel\n")

            if (row == (len(list_of_names) - 1)):
                if (row == column):
                    self.file_desc.write("unset ylabel\n")
                self.file_desc.write("set xlabel \'" + str(params_latex_dict[pair[1]]) + "\'\n")
                    
            if (column == 0):
                self.file_desc.write("set ylabel \'" + str(params_latex_dict[pair[0]]) + "\'\n")

            if (row < column):
                self.file_desc.write("set multiplot next\n\n")
                continue
            
            if (pair[0] == pair[1]):
                self.file_desc.write("set xrange  [" + str(params_lims_dict[pair[0]][0]) + ":" + str(params_lims_dict[pair[0]][1]) + "]\n")
                self.file_desc.write("set yrange [0:1.1]\n")

                filename = str(row) + "-1d.dat"
        
                pdf_ins = get_1d_pdf(params_dict[pair[0]], params_lims_dict[pair[0]][0], \
                                     params_lims_dict[pair[0]][1])
                pdf_ins.set_sigma_level()
                pdf_ins.prepare_histogram_from_subdivision(cut_off = 0, bins_number = 60)
                pdf_ins.prepare_get_interpolated_1d_distr()
                x_data = pdf_ins.x
                y_data = pdf_ins.get_interpolated_1d_distr(pdf_ins.x)
                below_zero = np.where(y_data < 0)
                y_data[below_zero] = 0.0 
                
                np.savetxt(filename, np.c_[x_data, y_data], fmt = '%12.5e')

                self.file_desc.write("plot \'" + str(filename) + "\' u 1:2 notitle with lines" \
                                 " ls 1 \n\n")
                
            else:
                self.file_desc.write("set xrange  [" + str(params_lims_dict[pair[1]][0]) + ":" + str(params_lims_dict[pair[1]][1]) + "]\n")
                self.file_desc.write("set yrange  [" + str(params_lims_dict[pair[0]][0]) + ":" + str(params_lims_dict[pair[0]][1]) + "]\n")

                filename = str(row) + "-vs-" + str(column) + "-2d.dat"

                contour_ins = get_2d_posterior(params_dict[pair[0]], params_dict[pair[1]])
                contour_ins.prepare_contour()

                np.savetxt(filename, np.c_[params_dict[pair[1]][::50], params_dict[pair[0]][::50]], fmt = '%12.5e')

                self.file_desc.write("plot \'" + str(filename) + "\' u 1:2 notitle w p" \
                                 " ls 2, \\\n")

                filename_1_sigma = str(row) + "-vs-" + str(column) + "-1-sigma.dat"
                filename_2_sigma = str(row) + "-vs-" + str(column) + "-2-sigma.dat"

                np.savetxt(filename_1_sigma, np.c_[contour_ins.y_vert_1_s, contour_ins.x_vert_1_s], fmt = '%12.5e')
                np.savetxt(filename_2_sigma, np.c_[contour_ins.y_vert_2_s, contour_ins.x_vert_2_s], fmt = '%12.5e')

                self.file_desc.write("\'" + str(filename_1_sigma) + "\' u 1:2 notitle w lines" \
                                 " ls 1, \\\n")

                self.file_desc.write("\'" + str(filename_2_sigma) + "\' u 1:2 notitle w lines" \
                                 " ls 1\n\n")

        self.file_desc.write("unset multiplot\n")
        self.file_desc.close()

    def plot(self):
        command = "gnuplot " + str(self.gnuplot_filename) + ".plt" \
                  " && latex " + str(self.gnuplot_filename) + ".tex > /dev/null" \
                  " && dvips " + str(self.gnuplot_filename) + ".dvi > /dev/null" \
                  " && ps2pdf " + str(self.gnuplot_filename) + ".ps > /dev/null"
        print(command)

        p = subprocess.Popen(['/bin/bash', '-c', command])
        output, errors = p.communicate()

##################################################
# построение графика для параметра Хаббла, вкладов
# в критическую плотность и т.п.
##################################################

hub = 0.7;
H_0 = hub / 2998;
N_eff = 3.04;
Omega_gamma = 2.469e-5 * pow(hub, -2.0);
Omega_r0 = Omega_gamma * (1 + 0.2271 * N_eff);

class SigmaCosmo(object):
    def __init__(self):
        self.Omega_m0 = mean_values_dict["Omega_m0"]
        self.Omega_DE0 = mean_values_dict["Omega_DE0"]
        self.tildeA = mean_values_dict["tildeA"]
        self.tildeB = mean_values_dict["tildeB"]
        self.Omega_curv0 = 1.0 - self.Omega_m0 - self.Omega_DE0 - Omega_r0 - self.tildeB
        self.nu = 2.0

    def atoH(self, a):
        atoH_value = self.Omega_m0 * a ** -3 + Omega_r0 * a ** -4 + \
                     self.Omega_DE0 - self.tildeA * np.log(a) + self.tildeB * a ** (- self.nu) +  \
                     self.Omega_curv0 * a ** -2;
        return np.sqrt(atoH_value)

    def atoOmegaDE(self, a):
        atoOmegaDE_value = (self.Omega_DE0 - self.tildeA * np.log(a) + self.tildeB * a ** - 2) / \
                           (self.atoH(a) ** 2)
        return atoOmegaDE_value
        

SigmaCosmo_ins = SigmaCosmo()
print(SigmaCosmo_ins.atoH(np.logspace(0.01, 0.1, 10)))
