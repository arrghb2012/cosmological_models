# -*- coding: utf-8 -*-
from __future__ import division
import pylab as P
import scipy.stats as stats
import numpy as np
from scipy import interpolate
import os
import cPickle

def pickle_results(filename=None, verbose=True):
    """Generator for decorator which allows pickling the results of a funcion

    Pickle is python's built-in object serialization.  This decorator, when
    used on a function, saves the results of the computation in the function
    to a pickle file.  If the function is called a second time with the
    same inputs, then the computation will not be repeated and the previous
    results will be used.

    This functionality is useful for computations which take a long time,
    but will need to be repeated (such as the first step of a data analysis).

    Parameters
    ----------
    filename : string (optional)
        pickle file to which results will be saved.
        If not specified, then the file is '<funcname>_output.pkl'
        where '<funcname>' is replaced by the name of the decorated function.
    verbose : boolean (optional)
        if True, then print a message to standard out specifying when the
        pickle file is written or read.

    Examples
    --------
    >>> @pickle_results('tmp.pkl', verbose=True)
    ... def f(x):
    ...     return x * x
    >>> f(4)
    @pickle_results: computing results and saving to 'tmp.pkl'
    16
    >>> f(4)
    @pickle_results: using precomputed results from 'tmp.pkl'
    16
    >>> f(6)
    @pickle_results: computing results and saving to 'tmp.pkl'
    36
    >>> import os; os.remove('tmp.pkl')
    """
    def pickle_func(f, filename=filename, verbose=verbose):
        if filename is None:
            filename = '%s_output.pkl' % f.__name__

        def new_f(*args, **kwargs):
            try:
                D = cPickle.load(open(filename, 'r'))
                cache_exists = True
            except:
                D = {}
                cache_exists = False

            # simple comparison doesn't work in the case of numpy arrays
            Dargs = D.get('args')
            Dkwargs = D.get('kwargs')

            try:
                args_match = (args == Dargs)
            except:
                args_match = np.all([np.all(a1 == a2)
                                     for (a1, a2) in zip(Dargs, args)])

            try:
                kwargs_match = (kwargs == Dkwargs)
            except:
                kwargs_match = ((sorted(Dkwargs.keys())
                                 == sorted(kwargs.keys()))
                                and (np.all([np.all(Dkwargs[key]
                                                    == kwargs[key])
                                             for key in kwargs])))

            if (type(D) == dict and D.get('funcname') == f.__name__
                    and args_match and kwargs_match):
                if verbose:
                    print ("@pickle_results: using precomputed "
                           "results from '%s'" % filename)
                retval = D['retval']

            else:
                if verbose:
                    print ("@pickle_results: computing results "
                           "and saving to '%s'" % filename)
                    if cache_exists:
                        print "  warning: cache file '%s' exists" % filename
                        print "    - args match:   %s" % args_match
                        print "    - kwargs match: %s" % kwargs_match
                retval = f(*args, **kwargs)
                cPickle.dump(dict(funcname=f.__name__, retval=retval,
                                  args=args, kwargs=kwargs),
                             open(filename, 'w'))
            return retval
        return new_f
    return pickle_func



def get_sig_levels(z):
	"""Get values corresponding to the different significance levels for a histo - argument is the histogrammed data"""

	linz = z.flatten()
	linz = np.sort(linz)[::-1]
	tot = sum(linz)

	acc = 0.0
	i=-1
	j=0
	lvls = [0.0, 0.68, 0.95, 0.997, 1.0, 1.01] # Significance levels (Gaussian)
	slevels = []
	for item in linz:
		acc += item/tot
		i+=1
		if(acc >= lvls[j]):
			j+=1
			slevels.append(item)
	return slevels[::-1]


def get_centroids(vals):
	"""Get bin centroids"""
	cent = []
	for i in range(len(vals)-1):
		cent.append(0.5*(vals[i]+vals[i+1]))
	return cent


class get_1d_posterior(object):
    # для 1d pdf используется весь интервал, который использовался для mcmc
    def __init__(self, cosmological_parameter, lowerbound, upperbound):
        self.cosmological_parameter = cosmological_parameter
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def set_sigma_level(self, sigma_level = 0.68):
        self.sigma_level = sigma_level

    def prepare_histogram_from_subdivision(self, cut_off = 1000, bins_number = 20):
        self.y, self.x = np.histogram(self.cosmological_parameter[cut_off:], bins = bins_number)
        self.x_new = np.zeros(len(self.x) - 1)
        for i in xrange(1, len(self.x) - 1):
            self.x_new[i - 1] = self.x[i - 1] + (self.x[i] - self.x[i - 1]) / 2.0
        self.x_new[-1] = self.x[-2] + (self.x[-1] - self.x[-2]) / 2.0
        self.y = self.y/float(self.y.max())
        self.x = self.x_new
        self.x_mean_pos = np.argmax(self.y)
        self.x_mean = self.x[self.x_mean_pos]

    def prepare_get_interpolated_1d_distr(self):
        self.interpolated_1d_distr = interpolate.splrep(self.x, self.y, s=0) # составление и запоминание
        # интерполяционной таблицы

    def get_interpolated_1d_distr(self, x):
        return interpolate.splev(x, self.interpolated_1d_distr, der = 0)

    def lhs_area(self, x):
        # площадь под кривой от левого края до значения x
        return quad(self.get_interpolated_1d_distr, self.x[0], x)[0]

    def rhs_area(self, x):
        # площадь под кривой от правого края до значения x
        return quad(self.get_interpolated_1d_distr, x, self.x[-1])[0]

    def fraction_for_lhs_sigma(self, x):
        return self.rhs_area(x) / self.rhs_area(self.x[0]) - ( - (1.0 - self.sigma_level) / 2.0 + 1.0) # according to trotta_handout

    def find_lhs_sigma(self, sigma_level = 0.68):
        lhs_sigma = bisect(self.fraction_for_lhs_sigma, self.x[0], self.x[-1])
        return lhs_sigma

    def fraction_for_rhs_sigma(self, x):
        return self.lhs_area(x) / self.lhs_area(self.x[-1]) - ( - (1.0 - self.sigma_level) / 2.0 + 1.0) # according to trotta_handout

    def find_rhs_sigma(self, sigma_level = 0.68):
        rhs_sigma = bisect(self.fraction_for_rhs_sigma, self.x[0], self.x[-1])
        return rhs_sigma


class get_2d_posterior(object):
    def __init__(self, cosmological_parameter_1, cosmological_parameter_2):
        self.cosmological_parameter_1 = cosmological_parameter_1
        self.cosmological_parameter_2 = cosmological_parameter_2

    def prepare_contour(self):
        z, x, y = np.histogram2d(self.cosmological_parameter_1, self.cosmological_parameter_2, bins = 20)
        cs = P.contour(get_centroids(x), get_centroids(y), z.T, get_sig_levels(z), colors = "black")
        P.close()
        p = cs.collections[- 1 - 1].get_paths()[0]
        v = p.vertices
        self.x_vert_1_s = v[:,0]
        self.y_vert_1_s = v[:,1]

        p = cs.collections[- 2 - 1].get_paths()[0]
        v = p.vertices
        self.x_vert_2_s = v[:,0]
        self.y_vert_2_s = v[:,1]


@pickle_results('tmp.pkl', verbose=True)
def prepare_data(burn_in, thinning_factor):
    cosmological_parameters = np.loadtxt('chains.dat', usecols = (0, 1, 2), unpack = True)
    new_size = np.int(np.ceil((cosmological_parameters.shape[1] - burn_in) / thinning_factor))
    new_cosmological_parameters = np.empty((cosmological_parameters.shape[0], new_size))
    for i in xrange(cosmological_parameters.shape[0]):
        new_cosmological_parameters[i] = cosmological_parameters[i][burn_in:][::thinning_factor]
    return new_cosmological_parameters

from scipy import interpolate
from scipy.optimize import bisect
from scipy.integrate import quad

class get_1d_pdf(object):
    def __init__(self, cosmological_parameter, lowerbound, upperbound):
        self.cosmological_parameter = cosmological_parameter
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def set_sigma_level(self, sigma_level = 0.68):
        self.sigma_level = sigma_level

    def prepare_histogram_from_subdivision(self, cut_off = 1000, bins_number = 20):
        self.y, self.x = np.histogram(self.cosmological_parameter[cut_off:], bins = bins_number)
        self.x_new = np.zeros(len(self.x) - 1)
        for i in xrange(1, len(self.x) - 1):
            self.x_new[i - 1] = self.x[i - 1] + (self.x[i] - self.x[i - 1]) / 2.0
        self.x_new[-1] = self.x[-2] + (self.x[-1] - self.x[-2]) / 2.0
        self.y = self.y/float(self.y.max())
        self.x = self.x_new
        self.x_mean_pos = np.argmax(self.y)
        self.x_mean = self.x[self.x_mean_pos]

    def prepare_histogram_from_subdivision_ext(self, cut_off = 1000, bins_number = 20):
        
        self.y, self.x = np.histogram(self.cosmological_parameter[cut_off:], bins = bins_number)
        self.x_new = np.zeros(len(self.x) - 1)
        for i in xrange(1, len(self.x) - 1):
            self.x_new[i - 1] = self.x[i - 1] + (self.x[i] - self.x[i - 1]) / 2.0
        self.x_new[-1] = self.x[-2] + (self.x[-1] - self.x[-2]) / 2.0
        self.y = self.y/float(self.y.max())
        self.x = self.x_new
        self.x_mean_pos = np.argmax(self.y)
        self.x_mean = self.x[self.x_mean_pos]

        hist_x_delta = self.x_new[1] - self.x_new[0]
        self.x_extended_lhs = [self.x_new[0] - hist_x_delta]
        while (self.x_extended_lhs[-1] - self.lowerbound) > 0:
            self.x_extended_lhs.append(self.x_extended_lhs[-1] - hist_x_delta) # последняя

        self.x_extended_rhs = [self.x_new[-1] + hist_x_delta]
        while (self.x_extended_rhs[-1] - self.upperbound) < 0:
            self.x_extended_rhs.append(self.x_extended_rhs[-1] + hist_x_delta) # последняя

        self.x = np.concatenate((np.array(self.x_extended_lhs)[::-1], 
                                np.concatenate((self.x, np.array(self.x_extended_rhs)))))

        self.y = np.concatenate((np.zeros_like(np.array(self.x_extended_lhs)[::-1]), 
                                np.concatenate((self.y, np.zeros_like(np.array(self.x_extended_rhs))))))

    def prepare_get_interpolated_1d_distr(self):
        self.interpolated_1d_distr = interpolate.splrep(self.x, self.y, s=0) # составление и запоминание
        # интерполяционной таблицы

    def get_interpolated_1d_distr(self, x):
        return interpolate.splev(x, self.interpolated_1d_distr, der = 0)

    def lhs_area(self, x):
        # площадь под кривой от левого края до значения x
        return quad(self.get_interpolated_1d_distr, self.x[0], x)[0]

    def rhs_area(self, x):
        # площадь под кривой от правого края до значения x
        return quad(self.get_interpolated_1d_distr, x, self.x[-1])[0]

    def fraction_for_lhs_sigma(self, x):
        return self.rhs_area(x) / self.rhs_area(self.x[0]) - ( - (1.0 - self.sigma_level) / 2.0 + 1.0) # according to trotta_handout

    def find_lhs_sigma(self, sigma_level = 0.68):
        lhs_sigma = bisect(self.fraction_for_lhs_sigma, self.x[0], self.x[-1])
        return lhs_sigma

    def fraction_for_rhs_sigma(self, x):
        return self.lhs_area(x) / self.lhs_area(self.x[-1]) - ( - (1.0 - self.sigma_level) / 2.0 + 1.0) # according to trotta_handout

    def find_rhs_sigma(self, sigma_level = 0.68):
        rhs_sigma = bisect(self.fraction_for_rhs_sigma, self.x[0], self.x[-1])
        return rhs_sigma
