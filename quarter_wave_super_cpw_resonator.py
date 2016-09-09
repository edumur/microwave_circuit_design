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

from super_CPW import SuperCPW

class QuarterWaveSuperCPWResonator(SuperCPW):



    def __init__(self, epsilon_r=11.68, tan_delta=7e-4, kappa=3.53e50,
                       w=19e-6, s=11.5e-6, t=100e-9, w_g=200e-6,
                       rho_n=3e-8, delta=180e-6, l=5e-3):
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
            Resistivity of the metal layer just above the transition in ohm.m.
        delta     : float
            Superconductor gap in eV.
        l         : float
            Length of the quarter wave resonator in meter.
        '''

        self.l = l

        SuperCPW.__init__(self, epsilon_r=epsilon_r, tan_delta=tan_delta,
                                    kappa=kappa, w=w, s=s, t=t,
                                    w_g=w_g,rho_n=rho_n, delta=delta)



    def get_resonator_capacitance(self):

        f0 = self.get_resonance_frequency()

        return self.l*self.get_capacitance_per_unit_length(f0)/2.



    def get_resonator_inductance(self):

        f0 = self.get_resonance_frequency()

        return 8.*self.l*self.get_inductance_per_unit_length(f0)/np.pi**2.



    def get_resonator_resistance(self):

        f0 = self.get_resonance_frequency()

        return self.get_characteristic_impedance(f0)/self.get_attenuation(f0)\
              /self.l



    def get_resonance_frequency(self, precision=4., Cc=None):
        '''
        Return the resonance frequency of a quarter wave resonator in Hz.

        Parameters
        ----------
        precision : {4} float
            Precision of the calculation, pretty useless since the convergence
            of the calculation is super fast.
        Cc : {None} float
            The coupling capacitance of the resonator in farad.
            If None return the resonance frequency of an uncoupled resonator.

        Return
        ----------
        f0 : float
            Resonance frequency of a quarterwave resonator.
        '''

        condition = 1.
        f0 = 8e9
        while condition>10.**-precision:

            l = self.get_inductance_per_unit_length(f0)
            c = self.get_capacitance_per_unit_length(f0)

            # If the coupling capacitance is not given, we return the resonance
            # frequency of an uncoupled resonator
            if Cc is None:
                f1 = np.pi/2./self.l/np.sqrt(l*c)/2./np.pi
            # If the coupling capacitance is given, we return the resonance
            # frequency of a coupled resonator
            else:
                f1 = np.pi/2./self.l/np.sqrt(l*c)/2./np.pi\
                     /(1. + Cc/self.l/c)

            condition = abs(f0 - f1)/1e9

            f0 = f1

        return f0



    def qc2cc(self, qc):
        '''
        Return the coupling capacitance from the coupling/external quality
        factor

        Parameters
        ----------
        qc : float
            Coupling quality factor

        Return
        ----------
        Cc : float
            The coupling capacitance of the resonator in farad.
        '''

        f0 = self.get_resonance_frequency()
        zr = self.get_characteristic_impedance(f0)
        o0 = f0*2.*np.pi

        return np.sqrt(np.pi/2./o0**2./qc/zr/50.)
