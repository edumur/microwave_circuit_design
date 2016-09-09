#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8

# Copyright (C) 2016 Dumur Étienne

# The kinetic inductance calculation comes from the following paper:
# W. Rauch, E. Gornik, G. Sölkner, A. A. Valenzuela, F. Fox and H. Behner
# Microwave properties of YBa2Cu3O7−x thin films studied with coplanar
# transmission line resonators
# Journal of Applied Physics, (1993), 73, 1866-1872
# doi: 10.1063/1.353173

# The penetration length calculation follows:
# R. L. Kautz
# Picosecond pulses on superconducting striplines
# Journal of Applied Physics 49, 308 (1978)
# doi: 10.1063/1.324387

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

import numpy as np
import scipy.constants as cst

from CPW import CPW

class SuperCPW(CPW):



    def __init__(self, epsilon_r = 11.68, tan_delta = 7e-4, kappa = 3.53e50,
                       w = 19e-6, s = 11.5e-6, t = 100e-9, w_g = 200e-6,
                       rho_n=3e-8, rrr = 2.4, delta=180e-6):
        '''

        Attributes
        ----------
        epsilon_r : float
            Relative permitivity of the substrat infarad per meter.
        tan_delta : float
            Loss tangent without dimension.
        kappa     : float
            Conductivity of the metal layer insiemens per meter.
        w         : float
            Width of the central line in meter.
        s         : float
            Width of the gap separation in meter.
        t         : float
            Thickness of the metal layer in meter.
        w_g       : float
            Width of the ground plane in meter.
        rho_n     : float
            Resistivity of the metal layer in ohm.m.
        rrr     : float
            Residual-resistance ratio of the metallic layer.
        delta     : float
            Superconductor gap in eV.
        '''

        self.rho_n = rho_n
        self.rrr   = rrr
        self.delta = delta

        CPW.__init__(self, epsilon_r=epsilon_r, tan_delta=tan_delta,kappa=kappa,
                           w=w, s=s, t=t, w_g=w_g)



    def __repr__(self):

        w_p, w_t = self._parse_number(self._w, 3)
        s_p, s_t = self._parse_number(self._s, 3)
        t_p, t_t = self._parse_number(self._t, 3)
        w_g_p, w_g_t = self._parse_number(self._w_g, 3)
        e_r_p, e_r_t = self._parse_number(self._epsilon_r, 3)
        kappa_p, kappa_t = self._parse_number(self._kappa, 3)
        rho_p, rho_t = self._parse_number(self.rho_n, 3)
        rrr_p, rrr_t = self._parse_number(self.rrr, 3)
        delta_p, delta_t = self._parse_number(self.delta, 3)

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
               '\n'\
               '    Superconductor parameters:\n'\
               '        Superconductor gap:     '+str(delta_p)+' '+str(delta_t)+'eV\n'\
               '        Normal resistivity:     '+rho_p+' '+rho_t+'ohm/m\n'\
               '        RRR:     '+str(rrr_p)+' '+str(rrr_t)+'\n'\



    def get_penetration_length(self):
        '''
        Return the london penetration length in m.
        '''

        return np.sqrt(cst.hbar*self.rho_n/self.rrr/cst.mu_0/np.pi/self.delta/cst.eV)




    def get_inductance_per_unit_length(self, f, separate=False):
        '''
        Return the length inductance of the transmision line by taking into
        account the superconductivity of the metal layer.

        Parameters
        ----------
        f : float, numpy.ndarray
            Frequency in Hz
        separate : Booleen
            If True return a tupple of inductance as (geometric, kinetic).

        Return
        ----------
        Ll : float, numpy.ndarray
            Inductance per unit length in H/m.
        '''

        Ll = CPW.get_inductance_per_unit_length(self, f)

        a = -self._t/np.pi +np.sqrt((2.*self._t/np.pi)**2. + self._w**2.)/2.
        b = self._w**2./4./a
        c = b - self._t/np.pi + np.sqrt((self._t/np.pi)**2. + self._s**2./2.)
        d = 2.*self._t/np.pi + c
        Lk = cst.mu_0*self.get_penetration_length()*c/4./a/d/self._ellipk(self._k0())*(\
              1.7/np.sinh(self._t/2./self.get_penetration_length())\
              + 0.4/np.sqrt(((b/a)**2. - 1.)*(1. - (b/d)**2.)))

        if separate:
            return Ll, Lk
        else:
            return Ll + Lk
