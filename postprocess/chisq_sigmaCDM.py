# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from run_kut5 import integrate
from scipy import interpolate
from math import pow

hub = 0.7;
H_0 = hub / 2998;
N_eff = 3.04;
Omega_gamma = 2.469e-5 * pow(hub, -2.0);
Omega_r = Omega_gamma * (1 + 0.2271 * N_eff);


class IB:
    s = 0
    ph = 1
    ph_pr = 2
    chi = 3
    chi_pr = 4
    dimB = 5

class SM(object):
    def __init__(self, lam, mu, barV):
        self.lam = lam
        self.mu = mu
        self.barV = barV

    def V(self, ph, chi):
        return self.barV*np.exp(-np.sqrt(self.lam)*ph)+self.barV*np.exp(-np.sqrt(self.lam)*chi)

    def dVdph(self, ph):
        return -np.sqrt(self.lam)*self.barV*np.exp(-np.sqrt(self.lam)*ph)

    def dVdchi(self, chi):
        return -np.sqrt(self.lam)*self.barV*np.exp(-np.sqrt(self.lam)*chi)

    def d2Vdph2(self, ph):
        return self.lam*self.barV*np.exp(-np.sqrt(self.lam)*ph)

    def d2Vdchi2(self, ph):
        return self.lam*self.barV*np.exp(-np.sqrt(self.lam)*ph)

    def d2Vdphdchi(self, ph, chi):
        return 0

    def h22(self, ph):
        return np.exp(np.sqrt(self.mu) * ph)

    def dh22dph(self, ph):
        return np.sqrt(self.mu) * np.exp(np.sqrt(self.mu) * ph)

    def dh22dchi(self, chi):
        return 0

    def d2h22dph2(self, ph):
        return self.mu*np.exp(np.sqrt(self.mu)*ph)

    def d2h22dchi2(self):
        return 0

    def d2h22dphidchi(self):
        return 0

class sigmaQCDM_model_background_w_rad(SM, object):
    def __init__(self, Omega_mi, Omega_DEi, a_i = 1.0e-6, omegaDE_i = -(1-1.0e-6), lam = 1, mu = 1):
        self.Omega_mi = Omega_mi
        self.Omega_DEi = Omega_DEi
        self.Omega_ri = 1 - Omega_mi - Omega_DEi
        self.a_i = a_i
        self.omegaDE_i = omegaDE_i
        self.barV = 3.0/4*(self.Omega_DEi)*(1-self.omegaDE_i)
        self.lam = lam
        self.mu = mu
        SM.__init__(self, self.lam, self.mu, self.barV)

    def background_system(self, x, y):
        dydx = np.zeros(IB.dimB)
        dydx[IB.s] = np.sqrt(y[IB.s]**2*(self.Omega_mi*y[IB.s]**(-3) + self.Omega_ri*y[IB.s]**(-4)\
                    + (0.5*y[IB.ph_pr]**2 + 0.5*SM.h22(self, y[IB.ph])*y[IB.chi_pr]**2 + SM.V(self, y[IB.ph], y[IB.chi]))/3))
        dydx[IB.ph] = y[IB.ph_pr]
        dydx[IB.ph_pr] = -3*dydx[IB.s]/y[IB.s]*y[IB.ph_pr] + 0.5*SM.dh22dph(self, y[IB.ph])*y[IB.chi_pr]**2 \
                         - SM.dVdph(self, y[IB.ph])
        dydx[IB.chi]=y[IB.chi_pr]
        dydx[IB.chi_pr] = - 3*dydx[IB.s]/y[IB.s]*y[IB.chi_pr] - 1.0 / SM.h22(self, y[IB.ph]) * SM.dh22dph(self, y[IB.ph]) \
                          *y[IB.ph_pr]*y[IB.chi_pr] - 1.0/SM.h22(self, y[IB.ph]) * SM.dVdchi(self, y[IB.chi])
        return dydx

    def set_ic_background(self):
        s_i = self.a_i / self.a_i
        ph_i = 0; chi_i = 0
        ph_pr_i = np.sqrt(3.0/2*(self.Omega_DEi)*(1+self.omegaDE_i))
        chi_pr_i = np.sqrt(3.0/2*(self.Omega_DEi)*(1+self.omegaDE_i))
        self.xInit = 0
        self.yInit = np.array([s_i, ph_i, ph_pr_i, chi_i, chi_pr_i])

    def solve_background(self):
        xStop = 5.0e+4
        yStop = 1.2 / self.a_i
        self.set_ic_background()
        self.BX, self.BY = integrate(self.background_system, \
                            self.xInit, self.yInit, xStop, yStop)
        return self.BX, self.BY

    def get_t(self):
        return self.BX

    def get_a(self):
        return self.a_i * self.BY[:, IB.s]

    def get_phi(self):
        return self.BY[:, IB.ph]

    def get_chi(self):
        return self.BY[:, IB.chi]

    def get_phi_dot(self):
        return self.BY[:, IB.ph_pr]

    def get_chi_dot(self):
        return self.BY[:, IB.chi_pr]

    def get_Omega_DE(self):
        Omega_DE = (0.5*(self.BY[:, IB.ph_pr])**2+0.5*SM.h22(self, self.BY[:, IB.ph])*self.BY[:, IB.chi_pr]**2+\
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi]))/3./(self.Omega_mi*(self.BY[:, IB.s])**(-3) \
                                                                           + self.Omega_ri*(self.BY[:, IB.s])**(-4) \
                    +(0.5*(self.BY[:, IB.ph_pr])**2+0.5*SM.h22(self, IB.ph)*self.BY[:, IB.chi_pr]**2+\
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi]))/3.0)

        return Omega_DE

    def get_Omega_r(self):
        Omega_r = (self.Omega_ri*(self.BY[:, IB.s])**(-4))/(self.Omega_mi*(self.BY[:, IB.s])**(-3) \
                                                            + self.Omega_ri*(self.BY[:, IB.s])**(-4) \
                    +(0.5*(self.BY[:, IB.ph_pr])**2+0.5*SM.h22(self, IB.ph)*self.BY[:, IB.chi_pr]**2+\
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi]))/3.0)
        return Omega_r

    def get_Omega_DE_of_a(self, a, interp_type = 'linear'):
        if interp_type == 'linear':
            f = interpolate.interp1d(self.a_i * self.BY[:, IB.s], self.get_Omega_DE())
            return f(a)
        elif interp_type == 'cubic':
            omega_DE1 = interpolate.splrep(self.a_i * self.BY[:, IB.s], self.get_Omega_DE(), s=0)
        return interpolate.splev(a, omega_DE1, der = 0)

    def get_Omega_r_of_a(self, a, interp_type = 'linear'):
        if interp_type == 'linear':
            f = interpolate.interp1d(self.a_i * self.BY[:, IB.s], self.get_Omega_r())
            return f(a)
        elif interp_type == 'cubic':
            omega_DE1 = interpolate.splrep(self.a_i * self.BY[:, IB.s], self.get_Omega_r(), s=0)
        return interpolate.splev(a, omega_DE1, der = 0)

    def get_Omega_m_of_a(self, a, interp_type = 'linear'):
        return 1.0 - self.get_Omega_DE_of_a(a, interp_type = 'linear') - self.get_Omega_r_of_a(a, interp_type = 'linear')

    def get_h22(self):
        h22 = SM.h22(self, self.BY[:, IB.ph])
        return h22

    def get_V(self):
        V = SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi])
        return V

    def DE_eq_of_state_param(self):
        omega_DE = (0.5 * (self.BY[:, IB.ph_pr]) ** 2 + 0.5*SM.h22(self, self.BY[:, IB.ph])*(self.BY[:, IB.chi_pr])**2 - \
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi])) / (0.5 * (self.BY[:, IB.ph_pr]) ** 2 +\
                    0.5*SM.h22(self, self.BY[:, IB.ph])*(self.BY[:, IB.chi_pr])**2 +\
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi]))
        return omega_DE

    def get_DE_eq_of_state_param_of_a(self, a, interp_type = 'linear'):
        if interp_type == 'linear':
            f = interpolate.interp1d(self.a_i * self.BY[:, IB.s], self.DE_eq_of_state_param())
            return f(a)
        elif interp_type == 'cubic':
            omega_DE1 = interpolate.splrep(self.a_i * self.BY[:, IB.s], self.DE_eq_of_state_param(), s=0)
        return interpolate.splev(a, omega_DE1, der = 0)

    def get_q(self):
        q = 0.5 * (self.Omega_mi*(self.BY[:, IB.s])**(-3) + 2 * self.Omega_ri * (self.BY[:, IB.s])**(-4) \
                   + (2 * (self.BY[:, IB.ph_pr]) ** 2 + \
                   2 * SM.h22(self, self.BY[:, IB.ph])*(self.BY[:, IB.chi_pr])**2 - \
                   2 * SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi])) / 3.0) / (self.Omega_mi*(self.BY[:, IB.s])**(-3) + \
                                                                                    self.Omega_ri*(self.BY[:, IB.s])**(-4) + \
                    +(0.5*(self.BY[:, IB.ph_pr])**2+0.5*SM.h22(self, IB.ph)*self.BY[:, IB.chi_pr]**2+ \
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi])) / 3.0)
        return q

    def get_omega_eff(self):
        omega_eff = ( 1.0/3.0 * self.Omega_ri * (self.BY[:, IB.s])**(-4) \
                   + (0.5 * (self.BY[:, IB.ph_pr]) ** 2 + \
                   0.5 * SM.h22(self, self.BY[:, IB.ph])*(self.BY[:, IB.chi_pr])**2 - \
                   SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi])) / 3.0) / (self.Omega_mi*(self.BY[:, IB.s])**(-3) + \
                                                                                self.Omega_ri*(self.BY[:, IB.s])**(-4) + \
                    +(0.5*(self.BY[:, IB.ph_pr])**2+0.5*SM.h22(self, IB.ph)*self.BY[:, IB.chi_pr]**2+ \
                    SM.V(self, self.BY[:, IB.ph], self.BY[:, IB.chi])) / 3.0)
        return omega_eff

    def get_q_of_a(self, a, interp_type = 'linear'):
        if interp_type == 'linear':
            f = interpolate.interp1d(self.a_i * self.BY[:, IB.s], self.get_q())
            return f(a)
        elif interp_type == 'cubic':
            omega_DE1 = interpolate.splrep(self.a_i * self.BY[:, IB.s], self.get_q(), s=0)
        return interpolate.splev(a, omega_DE1, der = 0)

    def get_s_pr0(self):
        S1 = np.sqrt(self.Omega_mi/self.BY[:, IB.s] + self.Omega_ri/self.BY[:, IB.s]**2 +self.BY[:, IB.s] ** 2 *(0.5*self.BY[:, IB.ph_pr]**2 + \
                0.5*SM.h22(self, self.BY[:, IB.ph])*self.BY[:, IB.chi_pr]**2+SM.V(self, self.BY[:, IB.ph], self.BY[:, IP.chi]))/3)
        f = interpolate.interp1d(self.BY[:, IP.s], S1)
        return f(1 / self.a_i)

    def get_H_of_a(self, a):       # с использованием self.DE_eq_of_state_param и interpolate.splint
        H_0 = 0.7 / 2998
        Omega_DE0 = self.get_Omega_DE_of_a(1.0)
        Omega_r0 = self.get_Omega_r_of_a(1.0)
        Omega_m0 = self.get_Omega_m_of_a(1.0)
        omega_DE1 = interpolate.splrep(self.a_i * self.BY[:, IB.s], (1 + self.DE_eq_of_state_param()) / (self.a_i * self.BY[:, IB.s]), s=0)
        integral = interpolate.splint(a, 1.0, omega_DE1)
        atoH = Omega_m0 * a **- 3 + Omega_r0 * a **- 4 + Omega_DE0 * np.exp(3 * integral)

        return atoH 

    def precomp_spline_for_H(self):       # с использованием self.DE_eq_of_state_param и interpolate.splint
        self.omega_DE1 = interpolate.splrep(self.a_i * self.BY[:, IB.s], (1 + self.DE_eq_of_state_param()) / (self.a_i * self.BY[:, IB.s]), s=0)

    def get_H_of_a_with_precomp_spline(self, a):       # с использованием self.DE_eq_of_state_param и interpolate.splint
        H_0 = 0.7 / 2998
        Omega_DE0 = self.get_Omega_DE_of_a(1.0)
        Omega_r0 = self.get_Omega_r_of_a(1.0)
        Omega_m0 = self.get_Omega_m_of_a(1.0)
        integral = interpolate.splint(a, 1.0, self.omega_DE1)
        atoH = Omega_m0 * a **- 3 + Omega_r0 * a **- 4 + Omega_DE0 * np.exp(3 * integral)

        return atoH
