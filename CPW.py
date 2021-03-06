#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8

# Based on an article of Wolfgang Heinrich
# "Quasi-TEM Description of MMIC coplanar Lines Including onductor-Loss
#  Effects"
# IEEE Transactions on Microwave Theory And Techniques, vol 41, n° 1,
# January 1993

# Copyright (C) 2013 Dumur Étienne

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import scipy.constants as cst
from scipy.special import ellipk, ellipkm1, ellipe
import numpy as np

class CPW(object):



    def __init__(self, epsilon_r = 11.68, tan_delta = 7e-4, kappa = 3.53e50,
                       w = 19e-6, s = 11.5e-6, t = 100e-9, w_g = 200e-6):
        '''Class allowing the calculation of the RLCG parameters for a
           coplanar waveguide (cpw).
            Based on an article of Wolfgang Heinrich
            "Quasi-TEM Description of MMIC coplanar Lines Including
            Conductor-Loss Effects"
            IEEE Transactions on Microwave Theory And Techniques, vol 41, n° 1,
            January 1993

            Input:
                - epsilon_r (float) : Relative permitivity of the substrat in
                                      farad per meter.
                - tan_delta (float) : Loss tangent without dimension.
                - kappa     (float) : Conductivity of the metal layer in
                                      siemens per meter.

                - w         (float) : Width of the central line in meter.
                - s         (float) : Width of the gap separation in meter.
                - t         (float) : Thickness of the metal layer in meter.
                - w_g       (float) : Width of the ground plane in meter.
        '''

        self._epsilon_r = epsilon_r
        self._tan_delta = tan_delta
        self._kappa     = kappa

        self._w   = w
        self._s   = s
        self._t   = t
        self._w_g = w_g

        self._a   = self._w/2.
        self._b   = self._w/2. + self._s
        self._t_H = self._t/2.

        # Limit above which the class doesn't use ellipk but ellipkm1
        self._ellipk_limit = 0.99



    def __repr__(self):

        w_p, w_t = self._parse_number(self._w, 3)
        s_p, s_t = self._parse_number(self._s, 3)
        t_p, t_t = self._parse_number(self._t, 3)
        w_g_p, w_g_t = self._parse_number(self._w_g, 3)
        e_r_p, e_r_t = self._parse_number(self._epsilon_r, 3)
        kappa_p, kappa_t = self._parse_number(self._kappa, 3)

        b = int(np.log10(self._tan_delta))
        c = self._tan_delta*10**-b

        return 'CoPlanar Waveguide instanced with following parameters:\n'\
               '\n'\
               '    Geometrical parameters:\n'\
               '        Central line width:      '+w_p+' '+w_t+'m\n'\
               '        Gap separation width:    '+s_p+' '+s_t+'m\n'\
               '        Thickness:               '+t_p+' '+t_t+'m\n'\
               '        Ground plane width:      '+w_g_p+' '+w_g_t+'m\n'\
               '\n'\
               '    Electrical parameters:\n'\
               '        Relative permitivity:    '+e_r_p+' '+e_r_t+'F/m\n'\
               '        Loss tangente:           '+str(c)+'e'+str(b)+'\n'\
               '        Electrical conductivity: '+kappa_p+' '+kappa_t+'S/m\n'\

    ##########################################################################
    #
    #
    #                                    Parameters
    #
    #
    ##########################################################################

    @property
    def conductivity(self):
        '''Conductivity of the metallic layer in siemens per meter.'''

        return self._kappa

    @conductivity.setter
    def conductivity(self, value):
        self._kappa = float(value)

    @property
    def loss_tangent(self):
        '''Loss tangent of the metallic layer without dimension.'''

        return self._tan_delta

    @loss_tangent.setter
    def loss_tangent(self, value):
        self._tan_delta = float(value)

    @property
    def relative_permittivity(self):
        '''Relative permittivity of the substrat layer in farad per meter.'''

        return self._epsilon_r

    @relative_permittivity.setter
    def relative_permittivity(self, value):
        self._epsilon_r = float(value)

    @property
    def width_ground_plane(self):
        '''Width of the coplanar ground plane in meter.'''

        return self._w_g

    @width_ground_plane.setter
    def width_ground_plane(self, value):
        self._w_g = float(value)

    @property
    def thickness(self):
        '''Thickness of the metal layer in meter.'''

        return self._t

    @thickness.setter
    def thickness(self, value):

        self._t = float(value)
        self._t_H = self._t/2.

    @property
    def width_gap_separation(self):
        '''Width of the gap separation in meter.'''

        return self._s

    @width_gap_separation.setter
    def width_gap_separation(self, value):

        self._s = float(value)
        self._b   = self._w/2. + self._s

    @property
    def width_central_line(self):
        '''Width of the central line in meter.'''

        return self._w

    @width_central_line.setter
    def width_central_line(self, value):

        self._w = float(value)
        self._a   = self._w/2.
        self._b   = self._w/2. + self._s

    ##########################################################################
    #
    #
    #                                    usefull function
    #
    #
    ##########################################################################

    def _omega(self, f):
        '''Return the angular frequency'''
        return 2.*cst.pi*f


    def _ellipk(self, k):
        '''
            Handle the calculation of the elliptic integral thanks to scipy
            special functions module.
            First we have to precise that with the scipy documention
            definition of ellipk or ellipkm1,
            the argument of these function are m = k**2.
            Next the current method will use ellipk or ellipkm1 following
            the value of m
        '''

        m = k**2.

        if m < self._ellipk_limit:
            return ellipk(m)
        else:
            return ellipkm1(m)

    ##########################################################################
    #
    #
    #                                    ki coefficients
    #
    #
    ##########################################################################

    def _k0(self):
        return self._w/(self._w + 2.*self._s)

    def _k1(self):
        return self._k0()\
              *np.sqrt((1. \
                        - ((self._w + 2.*self._s)\
                          /(self._w + 2.*self._s + 2*self._w_g))**2.)\
                        /(1. - (self._w\
                        /(self._w + 2.*self._s + 2*self._w_g))**2.))

    def _k2(self):
        return self._k0()*np.sqrt((1. \
                                   - ((self._w + 2.*self._s)\
                                      /(4.*self._w + 2.*self._s))**2.)\
                                  /(1. - (self._w\
                                          /(4.*self._w + 2.*self._s))**2.))

    ##########################################################################
    #
    #
    #                                    pci coefficients
    #
    #
    ##########################################################################

    def _pc0(self):
        return self._b\
               /(2.*self._a\
                *self._ellipk(np.sqrt(1. - self._k0()**2.))**2.)

    def _pc1(self):
        return 1. + np.log((8.*np.pi*self._a)/(self._a + self._b))\
                  + (self._a * np.log(self._b/self._a))/(self._a + self._b)

    def _pc2(self):
        return self._pc1()\
               - (2.*self._a\
                    *self._ellipk(np.sqrt(1. - self._k0()**2.))**2.)\
                 /self._b

    def _pc3(self):
        return (2.*self._b**2.*ellipe(1. - self._k0()**2.))\
               /(self._a*(self._b + self._a)\
                *self._ellipk(np.sqrt(1. - self._k0()**2.)))

    def _pc4(self):
        return ((self._b - self._a)/(self._b + self._a))\
               *(np.log((8.*np.pi*self._a)/(self._a + self._b))\
                 + self._a/self._b)

    def _pc5(self):
        return ((self._b - self._a)/(self._b + self._a))*np.log(3.)

    def _pc6(self):
        return ((self._b - self._a)/(self._b + self._a))\
               *np.log((24.*np.pi*self._b*(self._a + self._b))\
                       /((self._b - self._a)**2))\
               - (self._b*np.log(self._b/self._a))/(self._b + self._a)

    #################################################################################
    #
    #
    #                                    Inductance coefficients
    #
    #
    #################################################################################

    def _L_DC(self, w_1, w_2):
        return (cst.mu_0/(8.*np.pi))\
               *((4.*self._g_L(w_1))/w_1**2.\
                 + (1./w_2**2.)*(self._g_L(w_1 + 2.*self._s)\
                                 + self._g_L(w_1 + 2.*w_2 + 2.*self._s)\
                                 + 2.*self._g_L(w_2)\
                                 - 2.*self._g_L(w_1 + w_2 + 2.*self._s))\
                                 - 4./w_1/w_2*(self._g_L(w_1 + w_2 + self._s)\
                                               - self._g_L(w_1 + self._s)\
                                               + self._g_L(self._s)
                                               - self._g_L(w_2 + self._s)))

    def _g_L(self, x):
        return (self._t**2./12. - x**2./2.)*np.log(1. + (x/self._t)**2.)\
               + (x**4./12./self._t**2.)*np.log(1. + (self._t/x)**2.)\
               - (2.*x*self._t/3.)\
                 *np.arctan(x/self._t + (x/self._t)**2.)*np.arctan(self._t/x)

    def _L_1(self):
        return self._L_DC(self._w, 3.*self._w/2.) - cst.mu_0/4./self._F1()

    def _L_2(self):
        return np.sqrt(cst.mu_0/2./self._omega_L2()/self._kappa)\
               *(self._F_lc() + self._F_lg())/4./self._F_up(self._t/2.)**2.

    def _L_inf(self):
        return cst.mu_0/4./self._F_up(self._t/2.)

    ##########################################################################
    #
    #
    #                                    F coefficients
    #
    #
    ##########################################################################

    def _F_lc(self):
        if self._t_H <= self._s/2.:
            A = np.pi*self._b \
                + self._b*np.log(8.*np.pi*self._a/(self._a + self._b))\
                - (self._b - self._a)\
                  *np.log((self._b - self._a)/(self._b + self._a))\
                - self._b*np.log(2.*self._t_H/self._s)
            B = self._pc1()*self._pc3()\
                - self._pc2()\
                - self._b*self._pc4()/self._a\
                + self._pc5()\
                + (self._pc2() - self._pc3()
                   + self._b/self._a - 1. - self._pc5())\
                  *np.log(2.*self._t_H/self._s)
            C = self._pc3()*(1. - 3.*self._pc1()/2.)\
                + 3.*self._pc1()/2.\
                - 2.*self._pc2() + 1.\
                + 3.*self._b*self._pc4()/2./self._a\
                - self._b*(self._b - self._a)/self._a/(self._b + self._a)\
                + (2.*self._pc2()\
                   + self._pc1()*(self._pc3() - 1.)\
                   - self._b*self._pc4()/self._a)\
                  *np.log(2.*self._t_H/self._s)

            return self._pc0()/self._s\
                   *(A/(self._a + self._b)\
                     + self._t_H*B/self._s\
                     + (self._t_H/self._s)**2.*C)
        else:
            return 1./2./self._s\
                   + self._t_H/self._s**2\
                   + self._pc0()/self._s\
                     *(np.pi*self._b/(self._a + self._b)\
                       + self._pc6()/2.\
                       + 1./8.*(self._pc3()*(self._pc1() + 2.)\
                                - self._pc1()\
                                - self._b*self._pc4()/self._a\
                                - 2.*(self._a**2. + self._b**2)\
                                  /self._a/(self._a + self._b)))

    def _F_lg(self):
        if self._t_H <= self._s/2.:
            A = np.pi*self._a\
                + self._a*np.log(8.*np.pi*self._b/(self._b - self._a))\
                - self._b*np.log((self._b - self._a)/(self._b + self._a))\
                - self._a*np.log((2.*self._t_H)/self._s)
            B = self._a*self._pc1()*self._pc3()/self._b\
                + (1. - self._a/self._b)*self._pc1()\
                - self._pc2() - self._pc4() - self._pc5()\
                + (self._pc2() - self._a*self._pc3()/self._b\
                   + self._a/self._b - 1.\
                   + self._pc5())\
                  *np.log(2.*self._t_H/self._s)
            C = self._a*self._pc3()/self._b*(1. - 3.*self._pc1()/2.)\
                + 3.*self._a*self._pc1()/2./self._b\
                - 2.*self._pc2() + 2. - self._a/self._b +3.*self._pc4()/2.\
                - (self._b - self._a)/(self._b + self._a)\
                + (2.*self._pc2()\
                   + self._a*self._pc1()/self._b*(self._pc3() - 1.)\
                   - self._pc4())\
                  *np.log((2.*self._t_H)/self._s)

            return self._pc0()/self._s\
                   *(A/(self._a + self._b)\
                     + self._t_H*B/self._s\
                     + (self._t_H/self._s)**2.*C)
        else:
            return 1./2./self._s\
                   + self._t_H/self._s**2\
                   + self._pc0()/self._s\
                     *(np.pi*self._a/(self._a + self._b)\
                       + self._pc6()/2.\
                       + 1./8.*(- self._a*self._pc1()/self._b\
                                + self._a*self._pc3()/self._b\
                                  *(self._pc1() + 2.)\
                                - self._pc4()\
                                - 2.*(self._a**2 + self._b**2)\
                                  /self._b/(self._a + self._b)))

    def _F1(self):
        return self._F_up(self._t/2.)\
               + self._ellipk(self._k2())\
                              /self._ellipk(np.sqrt(1. - self._k2()**2.))\
               - self._ellipk(self._k1())\
                              /self._ellipk(np.sqrt(1. - self._k1()**2.))

    def _F_up(self, t):

        if t <= self._s/2. :
            return self._ellipk(self._k1())\
                   /self._ellipk(np.sqrt(1. - self._k1()**2.))\
                   + self._pc0()*(t/self._s*(self._pc1()
                                             - np.log(2.*t/self._s))\
                                  + (t/self._s)**2.\
                                     *(1.\
                                       - 3.*self._pc2()/2.\
                                       + self._pc2()*np.log(2.*t/self._s)))
        else :
            return self._ellipk(self._k1())\
                   /self._ellipk(np.sqrt(1. - self._k1()**2.))\
                   + (self._pc0()*(self._pc2() + 2.))/8. + t/self._s

    def _F_low(self):
        return self._ellipk(self._k1())\
               /self._ellipk(np.sqrt(1. - self._k1()**2.))

    ##########################################################################
    #
    #
    #                                    omega_Li coefficients
    #
    #
    ##########################################################################

    def _omega_L0(self):
        return 4./(cst.mu_0*self._kappa*self._t*self._w_g)

    def _omega_L1(self):
        return 4./(cst.mu_0*self._kappa*self._t*self._w)

    def _omega_L2(self):
        return 18./(cst.mu_0*self._kappa*self._t**2)

    ##########################################################################
    #
    #
    #                                    nu_i coefficients
    #
    #
    ##########################################################################

    def _nu_1(self):
        return np.log((self._L_DC(self._w, self._w_g) - self._L_inf())\
                      /self._L_1())\
               /np.log(self._omega_L0()/self._omega_L1())

    def _nu_2(self):
        return np.log(self._L_1()/self._L_2())\
               /np.log(self._omega_L1()/self._omega_L2())

    ##########################################################################
    #
    #
    #                                    eta_i coefficients
    #
    #
    ##########################################################################

    def _eta_1(self):
        return (self._w/self._w_g)**4*(self._nu_1()/(4. - self._nu_1()))

    def _eta_2(self):
        return (self._w/self._w_g)**2*(self._nu_1()/(4. - self._nu_1()))

    def _eta_3(self):
        return (2.*self._t/9./self._w)**3.\
               *(self._nu_2() - 1./2.)/(self._nu_2() + 5./2.)

    def _eta_4(self):
        return 2.*self._t/9./self._w\
               *(self._nu_2() + 1./2.)/(self._nu_2() + 5./2.)

    ##########################################################################
    #
    #
    #                                    a_iL coefficients
    #
    #
    ##########################################################################

    def _a_3L(self):
        return ((self._nu_2() - self._nu_1())\
                *(1. + self._eta_1())\
                *(1. - self._eta_4())\
                + 4.*self._eta_2()\
                + self._eta_4()*(1. - 3.*self._eta_1()))\
               /((self._nu_1() - self._nu_2())\
                *(1. + self._eta_1())\
                *(1. - self._eta_3())\
                + 4. - self._eta_3()*(1. - 3.*self._eta_1()))

    def _a_2L(self):
        return (1./(1. + self._eta_1()))\
               *(self._a_3L()\
                 *(1. - self._eta_3()) - self._eta_2() - self._eta_4())

    def _a_4L(self):
        return - 9./2.*self._w/self._t\
                *(self._eta_4() + self._a_3L()*self._eta_3())

    def _a_5L(self):
        return ((2.*self._t/9./self._w)**2)*self._a_3L() + self._a_4L()

    def _a_1L(self):
        return self._nu_1()/(4. - self._nu_1()) + self._eta_2()*self._a_2L()

    def _a_0L(self):
        return (1. - self._L_inf()/self._L_DC(self._w, self._w_g))\
               *(self._a_1L() + (self._w/self._w_g)**2.*self._a_2L())

    ##########################################################################
    #
    #
    #                                    omega_ci coefficients
    #
    #
    ##########################################################################

    def _omega_c1(self):
        return np.sqrt(2.)*4./cst.mu_0/self._kappa/self._t/self._w

    def _omega_c2(self):
        return 8./cst.mu_0/self._kappa\
               *((self._w + self._t)/self._w/self._t)**2.

    ##########################################################################
    #
    #
    #                                    omega_gi coefficients
    #
    #
    ##########################################################################

    def _omega_g1(self):
        return 2./cst.mu_0/self._kappa/self._t/self._w_g

    def _omega_g2(self):
        return 2./cst.mu_0/self._kappa\
               *((2.*self._w_g + self._t)/self._w_g/self._t)**2.

    ##########################################################################
    #
    #
    #                                    gamma coefficients
    #
    #
    ##########################################################################

    def _gamma_c(self):
        return (self._omega_c1()/self._omega_c2())**2.

    def _gamma_g(self):
        return (self._omega_g1()/self._omega_g2())**2.

    ##########################################################################
    #
    #
    #                                    R_ci coefficients
    #
    #
    ##########################################################################

    def _R_c0(self):
        return 1./self._kappa/self._w/self._t

    def _R_c1(self):
        return np.sqrt(self._omega_c2()*cst.mu_0/2./self._kappa)\
               *self._F_lc()/4./self._F_up(self._t/2.)**2.

    ##########################################################################
    #
    #
    #                                    R_gi coefficients
    #
    #
    ##########################################################################

    def _R_g0(self):
        return 1./2./self._kappa/self._w_g/self._t

    def _R_g1(self):
        return np.sqrt(self._omega_g2()*cst.mu_0/2./self._kappa)\
               *self._F_lg()/4./self._F_up(self._t/2.)**2.

    ##########################################################################
    #
    #
    #                                    nu coefficients
    #
    #
    ##########################################################################

    def _nu_c(self):
        return np.log(self._R_c0()/self._R_c1())\
               /np.log(self._omega_c1()/self._omega_c2())

    def _nu_g(self):
        return np.log(self._R_g0()/self._R_g1())\
               /np.log(self._omega_g1()/self._omega_g2())

    ##########################################################################
    #
    #
    #                                    a_ic coefficients
    #
    #
    ##########################################################################

    def _a_4c(self):
        return (self._gamma_c()*self._nu_c()\
                + 1./4.*(1./2. - self._nu_c())\
                  *(4. - self._nu_c()*(1 - self._gamma_c()**2.)))\
                /(4. - self._nu_c()\
                  - 1./4.*(1./2. - self._nu_c())\
                    *(4. - self._nu_c()*(1. - self._gamma_c()**2.)))

    def _a_3c(self):
        return 1./4.*(1./2. - self._nu_c())*(1. + self._a_4c())

    def _a_2c(self):
        return 1./self._gamma_c()*(self._a_4c() - self._a_3c())

    def _a_1c(self):
        return self._a_2c() + self._gamma_c()*self._a_3c()

    ##########################################################################
    #
    #
    #                                    a_ig coefficients
    #
    #
    ##########################################################################

    def _a_4g(self):
        return (self._gamma_g()*self._nu_g()\
                + 1./4.*(1./2. - self._nu_g())\
                  *(4. - self._nu_g()*(1 - self._gamma_g()**2.)))\
               /(4. - self._nu_g()
                - 1./4.*(1./2. - self._nu_g())\
                  *(4. - self._nu_g()*(1 - self._gamma_g()**2.)))

    def _a_3g(self):
        return 1./4.*(1./2. - self._nu_g())*(1. + self._a_4g())

    def _a_2g(self):
        return 1./self._gamma_g()*(self._a_4g() - self._a_3g())

    def _a_1g(self):
        return self._a_2g() + self._gamma_g()*self._a_3g()

    ##########################################################################
    #
    #
    #                                    R coefficients
    #
    #
    ##########################################################################

    def _Rc(self, f):

        f_a = f[f<self._omega_c1()/2./np.pi]
        f_b = f[f[f<self._omega_c2()/2./np.pi]>=self._omega_c1()/2./np.pi]
        f_c = f[f>=self._omega_c2()/2./np.pi]

        a = self._R_c0()\
               *(1. + self._a_1c()*(self._omega(f_a)/self._omega_c1())**2.)

        b = self._R_c1()\
               *(self._omega(f_b)/self._omega_c2())**self._nu_c()\
               *(1. + self._a_2c()*(self._omega_c1()/self._omega(f_b))**2.\
                    + self._a_3c()*(self._omega(f_b)/self._omega_c2())**2.)

        c = np.sqrt((self._omega(f_c)*cst.mu_0)/(2.*self._kappa))\
               *self._F_lc()/4./self._F_up(self._t/2.)**2\
               *(1. + self._a_4c()*(self._omega_c2()/self._omega(f_c))**2)

        return np.concatenate((a, b, c))


    def _Rg(self, f):

        f_a = f[f<self._omega_g1()/2./np.pi]
        f_b = f[f[f<self._omega_g2()/2./np.pi]>=self._omega_g1()/2./np.pi]
        f_c = f[f>=self._omega_g2()/2./np.pi]

        a = self._R_g0()\
               *(1. + self._a_1g()*(self._omega(f_a)/self._omega_g1())**2.)

        b = self._R_g1()\
               *(self._omega(f_b)/self._omega_g2())**self._nu_g()\
               *(1. + self._a_2g()*(self._omega_g1()/self._omega(f_b))**2\
                    + self._a_3g()*(self._omega(f_b)/self._omega_g2())**2)

        c = np.sqrt((self._omega(f_c)*cst.mu_0)/(2.*self._kappa))\
               *self._F_lg()/4./self._F_up(self._t/2.)**2\
               *(1. + self._a_4g()*(self._omega_g2()/self._omega(f_c))**2)

        return np.concatenate((a, b, c))

    ##########################################################################
    #
    #
    #                                    Final result
    #
    #
    ##########################################################################

    def get_inductance_per_unit_length(self, f):
        '''Return the length inductance of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Length inductance in Henrys per meter
        '''

        if not isinstance(f, np.ndarray):
            f = np.array([f])

        f_a = f[f<self._omega_L0()/2./np.pi]
        f_b = f[f[f<self._omega_L1()/2./np.pi]>=self._omega_L0()/2./np.pi]
        f_c = f[f[f<self._omega_L2()/2./np.pi]>=self._omega_L1()/2./np.pi]
        f_d = f[f>=self._omega_L2()/2./np.pi]

        a = self._L_DC(self._w, self._w_g)\
            *(1. + self._a_0L()*(self._omega(f_a)/self._omega_L0())**2.)

        b = self._L_inf()\
            + self._L_1()\
              *(self._omega(f_b)/self._omega_L1())**self._nu_1()\
              *(1. + self._a_1L()*(self._omega_L0()/self._omega(f_b))**2.\
                   + self._a_2L()*(self._omega(f_b)/self._omega_L1())**2.)

        c =  self._L_inf()\
             + self._L_2()\
               *(self._omega(f_c)/self._omega_L2())**self._nu_2()\
               *(1. + self._a_3L()*(self._omega_L1()/self._omega(f_c))**2.\
                    + self._a_4L()*(self._omega(f_c)/self._omega_L2())**2.)

        d = self._L_inf()\
            + np.sqrt(cst.mu_0/(2.*self._omega(f_d)*self._kappa))\
              *((self._F_lc() + self._F_lg())/4./self._F_up(self._t/2.)**2.)\
              *(1. + self._a_5L()*(self._omega_L2()/self._omega(f_d)))

        return np.concatenate((a, b, c, d))

    def get_resistance_per_unit_length(self, f):
        '''Return the length resistance of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Length resistance in Ohms per meter
        '''

        if not isinstance(f, np.ndarray):
            f = np.array([f])

        return self._Rc(f) + self._Rg(f)

    def get_capacitance_per_unit_length(self, f):
        '''Return the length capacitance of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Length capacitance in Farrad per meter
        '''

        temp = 2.*cst.epsilon_0*(self._F_up(self._t)\
                                 + self._epsilon_r*self._F_low())

        if type(f) is np.ndarray:
            return temp*np.ones_like(f)
        else:
            return temp

    def get_conductance_per_unit_length(self, f):
        '''Return the length conductance of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Length conductance in Siemens per meter
        '''

        return 2.*self._omega(f)*cst.epsilon_0\
               *self._epsilon_r*self._tan_delta*self._F_low()

    def get_characteristic_impedance(self, f, norm=True):
        '''Return the characteristic impedance of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Characteristic impedance in Ohms
        '''
        a = self.get_resistance_per_unit_length(f)\
            + 1j*self._omega(f)*self.get_inductance_per_unit_length(f)
        b = self.get_conductance_per_unit_length(f)\
            + 1j*self._omega(f)*self.get_capacitance_per_unit_length(f)

        if norm:
            return abs(np.sqrt(a/b))
        else:
            return np.sqrt(a/b)

    def get_complex_wave_vector(self, f):
        '''Return the real and imaginary part of the complex wave vector
        coefficient of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - (real, imag) : (float[np.ndarray], float|np.ndarray)
                                     real and imaginary value of the complex
                                     wave vector
        '''
        a = self.get_resistance_per_unit_length(f)\
            + 1j*self._omega(f)*self.get_inductance_per_unit_length(f)
        b = self.get_conductance_per_unit_length(f)\
            + 1j*self._omega(f)*self.get_capacitance_per_unit_length(f)

        c = np.sqrt(a*b)

        return c.real, c.imag


    def get_attenuation(self, f):
        '''Return the attenuation coefficient of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Attenuation coefficient
        '''

        r, i = self.get_complex_wave_vector(f)

        return r

    def get_wave_vector(self, f):
        '''Return the wave vector coefficient of the transmision line
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Wave vector
        '''

        r, i = self.get_complex_wave_vector(f)

        return i

    def get_velocity(self, f):
        '''Return the velocity of the wave in the coplanar wave guide
                - Input :
                    - Frequency (float | list | numpy.ndarray) in Hertz

                - Output :
                    - Velocity in unit of c (speed of light)
        '''

        return 1./np.sqrt(self.get_capacitance_per_unit_length(f)\
                         *self.get_inductance_per_unit_length(f))/cst.c



    ##########################################################################
    #
    #
    #                                    Print methods
    #
    #
    ##########################################################################

    def _parse_number(self, number, precision, inverse = False):

        power_ten = int(np.log10(number))/3*3

        if power_ten >= -24 and power_ten <= 18 :

            prefix = {-24 : 'y',
                      -21 : 'z',
                      -18 : 'a',
                      -15 : 'p',
                      -12 : 'p',
                       -9 : 'n',
                       -6 : 'µ',
                       -3 : 'm',
                        0 : '',
                        3 : 'k',
                        6 : 'M',
                        9 : 'G',
                       12 : 'T',
                       15 : 'p',
                       18 : 'E'}

            if inverse:
                return str(round(number*10.**-power_ten, precision)), prefix[-power_ten]
            else:
                return str(round(number*10.**-power_ten, precision)), prefix[power_ten]
        else:
            return str(round(number, precision)), ''


    def print_results(self, frequency, precision=3):
        '''
            Summarize all results of the CPW object.
        '''

        if type(frequency) is not float:
            raise ValueError('Frequency parameter should be float type')

        Ll   = self.get_inductance_per_unit_length(frequency)
        Ll_p, Ll_t = self._parse_number(Ll, precision)

        Cl = self.get_capacitance_per_unit_length(frequency)
        Cl_p, Cl_t = self._parse_number(Cl, precision)

        Rl = self.get_resistance_per_unit_length(frequency)
        Rl_p, Rl_t = self._parse_number(Rl, precision)

        Gl = self.get_conductance_per_unit_length(frequency)
        Gl_p, Gl_t = self._parse_number(Gl, precision)

        a = self.get_attenuation(frequency)
        a_p, a_t = self._parse_number(a, precision, inverse=True)

        w = self.get_wave_vector(frequency)
        w_p, w_t = self._parse_number(w, precision, inverse=True)

        v = self.get_velocity(frequency)
        v_p, v_t = self._parse_number(v, precision)

        z = self.get_characteristic_impedance(frequency)
        z_p, z_t = self._parse_number(z, precision)

        print '------------------------------------------------------------'
        print '            Results'
        print ''
        print '    Inductance per unit length:    '+Ll_p+' '+Ll_t+'H/m'
        print '    Capacitance per unit length:   '+Cl_p+' '+Cl_t+'F/m'
        print '    Resistance per unit length:    '+Rl_p+' '+Rl_t+'Ω/m'
        print '    Conductance per unit length:   '+Gl_p+' '+Gl_t+'S/m'
        print ''
        print '    Attenuation:            '+a_p+' /'+a_t+'m'
        print '    Wave vector:            '+w_p+' /'+w_t+'m'
        print '    Velocity:            '+v_p+' /'+v_t+'c'
        print '    Characteristic impedance:    '+z_p+' /'+z_t+'Ω'
        print '------------------------------------------------------------'
