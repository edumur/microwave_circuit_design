#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8

# Copyright (C) 2016 Dumur Étienne

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

class Super_CPW(CPW):



    def __init__(self, epsilon_r = 11.68, tan_delta = 7e-4, kappa = 3.53e50,
                       w = 19e-6, s = 11.5e-6, t = 100e-9, w_g = 200e-6,
                       rho_n = 0.025e-8, delta=210e-6):
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
            Resistivity of the metal layer just above the transition in ohm.cm.
        delta     : float
            Superconductor gap in eV.
        '''

        self.rho_n = rho_n
        self.delta = delta

        CPW.__init__(self, epsilon_r=epsilon_r, tan_delta=tan_delta,kappa=kappa,
                           w=w, s=s, t=t, w_g=w_g)



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

        k = self._ellipk(self._k1())**2.
        d = self._t/4./np.pi/np.exp(np.pi)
        g = (self._w + 2.*self._s)**2./32/k/self._s/(self._w+self._s)\
            *(  2.*np.log(self._w*self._s/d/(self._w+self._s))\
                  /self._w\
              + 2.*np.log((self._w+2*self._s)*self._s/d/(self._w+self._s))\
                  /(self._w+2*self._s))

        lambda_eff = np.sqrt(cst.hbar*self.rho_n/cst.mu_0/np.pi/self.delta/cst.eV)
        Lk = cst.mu_0*lambda_eff*g/np.tanh(self._t/lambda_eff)

        if separate:
            return Ll, Lk
        else:
            return Ll + Lk
