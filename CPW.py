# -*- coding: utf-8 -*-

#Based on an article of Wolfgang Hainrich
#"Quasi-TEM Description of MMIC coplanar Lines Including onductor-Loss Effects"
# IEEE Transactions on Microwave Theory And Techniques, vol 41, n° 1, January 1993

#Copyright (C) 2013 Dumur Étienne

#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 2 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along
#with this program; if not, write to the Free Software Foundation, Inc.,
#51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import scipy.constants as cst
from scipy.special import ellipk, ellipe
from scipy.optimize import fsolve
import numpy as np

class CPW():
	
	def __init__(self, epsilon_r = 11.68, tan_delta = 7e-4, kappa = 3.53e50, w = 19e-6, s = 11.5e-6, t = 100e-9, w_g = 200e-6):
		'''Class allowing the calculation of the parameters of a coplanar waveguide.
			
			Input:
				- epsilon_r (float) : Relative permitivity of the substrat in farad per meter.
				- tan_delta (float) : Loss tangent.
				- kappa     (float) : Conductivity of the metal layer in Siemens per meter.
				
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
		
#		self.lambda_0 = 40e-9
	
	#################################################################################
	#
	#
	#									Set parameters
	#
	#
	#################################################################################
	
	def set_conductivity(self, kappa):
		'''
			Set the conductivity of the metallic layer.
			
			Input:
				- kappa (float):Conductivity of the metallic layer in siemens per meter.
		'''
		
		self._kappa = float(kappa)
	
	def set_loss_tangent(self, tan_delta):
		'''
			Set the loss tangent of the metallic layer.
			
			Input:
				- tan_delta (float): Loss tangent of the metallic layer.
		'''
		
		self._tan_delta = float(tan_delta)
	
	def set_relative_permittivity(self, epsilon_r):
		'''
			Set the epsilon_r of the substrat layer.
			
			Input:
				- epsilon_r (float): Epsilon_r of the substrat layer in farad per meter.
		'''
		
		self._epsilon_r = float(epsilon_r)
	
	def set_width_ground_plane(self, w_g):
		'''
			Set the thickness of the metal layer.
			
			Input:
				- w (float): Thickness of the metal layer in meter.
		'''
		
		self._w_g = float(w_g)
	
	def set_thickness(self, t):
		'''
			Set the thickness of the metal layer.
			
			Input:
				- w (float): Thickness of the metal layer in meter.
		'''
		
		self._t = float(t)
		self._t_H = self._t/2.
	
	def set_width_gap_separation(self, s):
		'''
			Set the width of the gap separation.
			
			Input:
				- s (float): Width of the gap separation in meter.
		'''
		
		self._s = float(s)
		self._b   = self._w/2. + self._s
	
	def set_width_central_line(self, w):
		'''
			Set the width of the central line.
			
			Input:
				- w (float): Width of the central line in meter.
		'''
		
		self._w = float(w)
		self._a   = self._w/2.
		self._b   = self._w/2. + self._s
	
	
	#################################################################################
	#
	#
	#									usefull function
	#
	#
	#################################################################################
	
	def _omega(self, f):
		'''Return the impulsion to a frequency'''
		return 2.*cst.pi*f
	
	
	#################################################################################
	#
	#
	#									Kinetic inductance calculation
	#
	#
	#################################################################################
#	def _g(self):
#		return (1./( 2*self._k0()**2*ellipk(self._k0())**2 ))*( -np.log(self._t/(4.*self._w)) - (self._w/(self._w + 2.*self._s)) * np.log(self._t/(4.*(self._w + 2.*self._s))) + ((2.*(self._w+self._s))/(self._w + 2.*self._s))*np.log(self._s/(self._w+self._s)) )
#	
#	def L_k(self):
#		return cst.mu_0*self.lambda_0**2*self._g()/(self._t*self._w)
	
	#################################################################################
	#
	#
	#									ki coefficients
	#
	#
	#################################################################################
	
	def _k0(self):
		return self._w / (self._w + 2.*self._s)
	
	def _k1(self):
		return self._k0()*np.sqrt((1. - ((self._w+2.*self._s)/(self._w+2.*self._s+2*self._w_g))**2)/(1-((self._w)/(self._w+2.*self._s+2*self._w_g))**2))
	
	def _k2(self):
		return self._k0()*np.sqrt((1-((self._w + 2.*self._s)/(4.*self._w +2.*self._s))**2 )/( 1 - ((self._w)/(4.*self._w + 2*self._s))**2))
	
	#################################################################################
	#
	#
	#									pci coefficients
	#
	#
	#################################################################################
	
	def _pc0(self):
		return self._b/(2.*self._a*ellipk(np.sqrt(1 - self._k0()**2))**2.)
	
	def _pc1(self):
		return 1. + np.log((8.*np.pi*self._a)/(self._a + self._b)) + (self._a * np.log(self._b/self._a))/(self._a + self._b)
	
	def _pc2(self):
		return self._pc1() - (2.*self._a*ellipk(np.sqrt(1 - self._k0()**2))**2.)/self._b
	
	def _pc3(self):
		return (2*self._b**2*ellipe(np.sqrt(1-self._k0()**2)))/(self._a*(self._b + self._a)*ellipk(np.sqrt(1-self._k0()**2)))
	
	def _pc4(self):
		return ((self._b-self._a)/(self._b+self._a))*(np.log((8.*np.pi*self._a)/(self._a + self._b)) + self._a/self._b)
	
	def _pc5(self):
		return ((self._b-self._a)/(self._b+self._a))*np.log(3)
	
	def _pc6(self):
		return ((self._b-self._a)/(self._b+self._a))*np.log((24.*np.pi*self._b*(self._a+self._b))/((self._b-self._a)**2)) - (self._b*np.log(self._b/self._a))/(self._b+self._a)
	
	#################################################################################
	#
	#
	#									Inductance coefficients
	#
	#
	#################################################################################
	
	def _L_DC(self, w_1, w_2):
		return (cst.mu_0/(8.*np.pi))*((4.*self._g_L(w_1))/w_1**2 + (1./w_2**2)*(self._g_L(w_1 + 2.*self._s) + self._g_L(w_1 + 2*w_2 +2.*self._s) + 2.*self._g_L(w_2) - 2.*self._g_L(w_1 + w_2 + 2.*self._s)) - (4./(w_1*w_2))*(self._g_L(w_1+w_2+self._s) - self._g_L(w_1+self._s) +self._g_L(self._s) - self._g_L(w_2+self._s)))
	
	def _g_L(self, x):
		return (self._t**2/12. - x**2/2.)*np.log(1 + (x/self._t)**2) + (x**4/(12*self._t**2))*np.log(1 + (self._t/x)**2) - ((2.*x*self._t)/3.)*(np.arctan(x/self._t) + (x/self._t)**2*np.arctan(self._t/x))
	
	def _L_1(self):
		return self._L_DC(self._w, (3.*self._w)/2.) - cst.mu_0/(4.*self._F1())
	
	def _L_2(self):
		return np.sqrt(cst.mu_0/(2*self._omega_L2()*self._kappa))*((self._F_lc() + self._F_lg())/(4.*self._F_up(self._t/2.)**2))
	
	def _L_inf(self):
		return cst.mu_0/(4.*self._F_up(self._t/2.))
	
	#################################################################################
	#
	#
	#									F coefficients
	#
	#
	#################################################################################
	
	def _F_lc(self):
		if self._t_H<= self._s/2.:
			A = np.pi*self._b + self._b*np.log((8.*np.pi*self._a)/(self._a + self._b)) - (self._b - self._a)*np.log((self._b-self._a)/(self._b+self._a)) - self._b*np.log((2*self._t_H)/self._s)
			B = self._pc1()*self._pc3() - self._pc2() - self._b*self._pc4()/self._a + self._pc5() + (self._pc2() -self._pc3() + self._b/self._a - 1. - self._pc5())*np.log((2.*self._t_H)/self._s)
			C = self._pc3()*(1.-3.*self._pc1()/2.) + 3.*self._pc1()/2. -2.*self._pc2() + 1. +(3.*self._b*self._pc4())/(2.*self._a) - (self._b*(self._b-self._a))/(self._a*(self._b+self._a)) + (2.*self._pc2() + self._pc1()*(self._pc3() - 1.) - self._b*self._pc4()/self._a)*np.log((2.*self._t_H)/self._s)
			
			return (self._pc0()/self._s)*(A/(self._a+self._b) + self._t_H*B/self._s + (self._t_H/self._s)**2*C)
		else:
			return 1./(2.*self._s) + self._t_H/self._s**2 + (self._pc0()/self._s)*((np.pi*self._b)/(self._a+self._b) +self._pc6()/2. +(1./8.)*(-self._pc1() +self._pc3()*(self._pc1()+2.) -self._b*self._pc4()/self._a -(2.*(self._a**2 + self._b**2))/(self._a*(self._a+self._b))))
	
	def _F_lg(self):
		if self._t_H<= self._s/2.:
			A = np.pi*self._a + self._a*np.log((8.*np.pi*self._b)/(self._b-self._a)) - self._b *np.log((self._b-self._a)/(self._b+self._a)) - self._a*np.log((2*self._t_H)/self._s)
			B = (self._a*self._pc1()*self._pc3())/self._b +(1.-self._a/self._b)*self._pc1() - self._pc2() - self._pc4() - self._pc5() + (self._pc2() - self._a*self._pc3()/self._b +self._a/self._b - 1. +self._pc5())*np.log((2.*self._t_H)/self._s)
			C = ((self._a*self._pc3())/self._b)*(1.-3.*self._pc1()/2.) + (3.*self._a*self._pc1())/(2.*self._b) -2.*self._pc2() +2. - self._a/self._b +3.*self._pc4()/2. - (self._b-self._a)/(self._b+self._a) + (2.*self._pc2() + ((self._a*self._pc1())/self._b)*(self._pc3() - 1.) - self._pc4())*np.log((2.*self._t_H)/self._s)
		
			return (self._pc0()/self._s)*(A/(self._a+self._b) + self._t_H*B/self._s + (self._t_H/self._s)**2*C)
		else:
			return 1./(2.*self._s) + self._t_H/self._s**2 + (self._pc0()/self._s)*((np.pi*self._a)/(self._a+self._b) +self._pc6()/2. +(1./8.)*(-(self._a*self._pc1())/self._b +((self._a*self._pc3())/self._b)*(self._pc1()+2.) - self._pc4() -(2.*(self._a**2 + self._b**2))/(self._b*(self._a+self._b))))
	
	def _F1(self):
		return self._F_up(self._t/2.) + ellipk(self._k2())/ellipk(np.sqrt(1-self._k2()**2)) - ellipk(self._k1())/ellipk(np.sqrt(1-self._k1()**2))
	
	def _F_up(self, t):
		
		if t<= self._s/2. :
			return ellipk(self._k1())/ellipk(np.sqrt(1-self._k1()**2)) + self._pc0()*( (t/self._s)*(self._pc1() - np.log((2.*t)/self._s)) +(t/self._s)*(t/self._s)*(1. - (3.*self._pc2())/2. + self._pc2()*np.log((2.*t)/self._s)))
		else :
			return ellipk(self._k1())/ellipk(np.sqrt(1-self._k1()**2)) + (self._pc0()*(self._pc2() + 2.))/8. + t/self._s
	
	def _F_low(self):
		return ellipk(self._k1())/ellipk(np.sqrt(1 - self._k1()**2))
	
	#################################################################################
	#
	#
	#									omega_Li coefficients
	#
	#
	#################################################################################
	
	def _omega_L0(self):
		return 4./(cst.mu_0*self._kappa*self._t*self._w_g)
	
	def _omega_L1(self):
		return 4./(cst.mu_0*self._kappa*self._t*self._w)
	
	def _omega_L2(self):
		return 18./(cst.mu_0*self._kappa*self._t**2)
	
	#################################################################################
	#
	#
	#									nu_i coefficients
	#
	#
	#################################################################################
	
	def _nu_1(self):
		return np.log((self._L_DC(self._w, self._w_g) - self._L_inf())/self._L_1())/np.log(self._omega_L0()/self._omega_L1())
	
	def _nu_2(self):
		return np.log(self._L_1()/self._L_2())/np.log(self._omega_L1()/self._omega_L2())
	
	#################################################################################
	#
	#
	#									eta_i coefficients
	#
	#
	#################################################################################
	
	def _eta_1(self):
		return (self._w/self._w_g)**4*(self._nu_1()/(4.- self._nu_1()))
	
	def _eta_2(self):
		return (self._w/self._w_g)**2*(self._nu_1()/(4.- self._nu_1()))
	
	def _eta_3(self):
		return (((2.*self._t)/(9*self._w))**3)*((self._nu_2() - 1./2.)/(self._nu_2() + 5./2.))
	
	def _eta_4(self):
		return ((2.*self._t)/(9*self._w))*((self._nu_2() + 1./2.)/(self._nu_2() + 5./2.))
	
	#################################################################################
	#
	#
	#									a_iL coefficients
	#
	#
	#################################################################################
	
	def _a_3L(self):
		return ((self._nu_2() - self._nu_1())*(1. + self._eta_1())*(1. - self._eta_4()) + 4.*self._eta_2() + self._eta_4()*(1. - 3.*self._eta_1()))/((self._nu_1() - self._nu_2())*(1. + self._eta_1())*(1. - self._eta_3()) + 4. -self._eta_3()*(1. - 3.*self._eta_1()))
	
	def _a_2L(self):
		return (1./(1. + self._eta_1()))*( self._a_3L() * (1. - self._eta_3()) - self._eta_2() - self._eta_4() )
	
	def _a_4L(self):
		return -(9./2.)*(self._w/self._t)*(self._eta_4() + self._a_3L()*self._eta_3())
	
	def _a_5L(self):
		return (((2.*self._t)/(9.*self._w))**2)*self._a_3L() + self._a_4L()
	
	def _a_1L(self):
		return self._nu_1()/(4. - self._nu_1()) + self._eta_2()*self._a_2L()
	
	def _a_0L(self):
		return (1. - self._L_inf()/self._L_DC(self._w, self._w_g))*(self._a_1L() + (self._w/self._w_g)**2*self._a_2L())
	
	#################################################################################
	#
	#
	#									omega_ci coefficients
	#
	#
	#################################################################################
	
	def _omega_c1(self):
		
		return np.sqrt(2.)*(4./(cst.mu_0*self._kappa*self._t*self._w))
	
	def _omega_c2(self):
		return (8./(cst.mu_0*self._kappa))*((self._w + self._t)/(self._w*self._t))**2.
	
	#################################################################################
	#
	#
	#									omega_gi coefficients
	#
	#
	#################################################################################
	
	def _omega_g1(self):
		return 2./(cst.mu_0*self._kappa*self._t*self._w_g)
	
	def _omega_g2(self):
		return (2./(cst.mu_0*self._kappa))*((2*self._w_g + self._t)/(self._w_g*self._t))**2.
	
	#################################################################################
	#
	#
	#									gamma coefficients
	#
	#
	#################################################################################
	
	def _gamma_c(self):
		return (self._omega_c1()/self._omega_c2())**2
	
	def _gamma_g(self):
		return (self._omega_g1()/self._omega_g2())**2
	
	#################################################################################
	#
	#
	#									R_ci coefficients
	#
	#
	#################################################################################
	
	def _R_c0(self):
		return 1./(self._kappa*self._w*self._t)
	
	def _R_c1(self):
		return np.sqrt((self._omega_c2() * cst.mu_0)/(2.*self._kappa))*(self._F_lc()/(4.*self._F_up(self._t/2.)**2.))
	
	#################################################################################
	#
	#
	#									R_gi coefficients
	#
	#
	#################################################################################
	
	def _R_g0(self):
		return 1./(2.*self._kappa*self._w_g*self._t)
	
	def _R_g1(self):
		return np.sqrt((self._omega_g2() * cst.mu_0)/(2.*self._kappa))*(self._F_lg()/(4.*self._F_up(self._t/2.)**2.))
	
	#################################################################################
	#
	#
	#									nu coefficients
	#
	#
	#################################################################################
	
	def _nu_c(self):
		return np.log(self._R_c0()/self._R_c1())/np.log(self._omega_c1()/self._omega_c2())
	
	def _nu_g(self):
		return np.log(self._R_g0()/self._R_g1())/np.log(self._omega_g1()/self._omega_g2())
	
	#################################################################################
	#
	#
	#									a_ic coefficients
	#
	#
	#################################################################################
	
	def _a_4c(self):
		return (self._gamma_c()*self._nu_c() + (1./4.)*(1./2. - self._nu_c())*(4. - self._nu_c()*(1 - self._gamma_c()**2)))/(4. - self._nu_c() - (1./4.)*(1./2. - self._nu_c())*(4. - self._nu_c()*(1 - self._gamma_c()**2)))
	
	def _a_3c(self):
		return (1./4.)*(1./2. - self._nu_c())*(1. + self._a_4c())
	
	def _a_2c(self):
		return (1./self._gamma_c())*(self._a_4c() - self._a_3c())
	
	def _a_1c(self):
		return self._a_2c() + self._gamma_c()*self._a_3c()
	
	#################################################################################
	#
	#
	#									a_ig coefficients
	#
	#
	#################################################################################
	
	def _a_4g(self):
		return (self._gamma_g()*self._nu_g() + (1./4.)*(1./2. - self._nu_g())*(4. - self._nu_g()*(1 - self._gamma_g()**2)))/(4. - self._nu_g() - (1./4.)*(1./2. - self._nu_g())*(4. - self._nu_g()*(1 - self._gamma_g()**2)))
	
	def _a_3g(self):
		return (1./4.)*(1./2. - self._nu_g())*(1. + self._a_4g())
	
	def _a_2g(self):
		return (1./self._gamma_g())*(self._a_4g() - self._a_3g())
	
	def _a_1g(self):
		return self._a_2g() + self._gamma_g()*self._a_3g()
	
	#################################################################################
	#
	#
	#									R coefficients
	#
	#
	#################################################################################
	
	def _Rc(self, f):
		first_condition = np.ma.masked_less_equal(f, self._omega_c1()).mask
		second_condition = np.ma.masked_less_equal(f, self._omega_c2()).mask
		
		if first_condition:
			return self._R_c0()*(1. + self._a_1c()*(self._omega(f)/self._omega_c1())**2)
		elif [~first_condition] and second_condition:
			return self._R_c1()*(self._omega(f)/self._omega_c2())**self._nu_c()*(1. + self._a_2c()*(self._omega_c1()/self._omega(f))**2 + self._a_3c()*(self._omega(f)/self._omega_c2())**2)
		elif [~second_condition]:
			return np.sqrt((self._omega(f)*cst.mu_0)/(2.*self._kappa))*(self._F_lc()/(4.*self._F_up(self._t/2.)**2))*(1. + self._a_4c()*(self._omega_c2()/self._omega(f))**2)
		
	
	def _Rg(self, f):
		first_condition = np.ma.masked_less_equal(f, self._omega_g1()).mask
		second_condition = np.ma.masked_less_equal(f, self._omega_g2()).mask
		
		if first_condition:
			return self._R_g0()*(1. + self._a_1g()*(self._omega(f)/self._omega_g1())**2)
		elif [~first_condition] and second_condition:
			return self._R_g1()*(self._omega(f)/self._omega_g2())**self._nu_g()*(1. + self._a_2g()*(self._omega_g1()/self._omega(f))**2 + self._a_3g()*(self._omega(f)/self._omega_g2())**2)
		elif  [~second_condition]:
			return np.sqrt((self._omega(f)*cst.mu_0)/(2.*self._kappa))*(self._F_lg()/(4.*self._F_up(self._t/2.)**2))*(1. + self._a_4g()*(self._omega_g2()/self._omega(f))**2)
		
	#################################################################################
	#
	#
	#									Final result
	#
	#
	#################################################################################
	
	def get_inductance_per_unit_length(self, f):
		'''Return the length inductance of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Length inductance in Henrys per meter
		'''
		
		first_condition = np.ma.masked_less_equal(f, self._omega_L0()).mask
		second_condition = np.ma.masked_less_equal(f, self._omega_L1()).mask
		third_condition = np.ma.masked_less_equal(f, self._omega_L2()).mask
		
		if first_condition:
			return self._L_DC(self._w, self._w_g)*(1. + self._a_0L()*(self._omega(f)/self._omega_L0())**2.)
		elif [~first_condition] and second_condition :
			return self._L_inf() + self._L_1()*(self._omega(f)/self._omega_L1())**self._nu_1()*(1. + self._a_1L()*(self._omega_L0()/self._omega(f))**2. + self._a_2L()*(self._omega(f)/self._omega_L1())**2. )
		elif [~second_condition] and third_condition :
			return self._L_inf() +self._L_2()*(self._omega(f)/self._omega_L2())**self._nu_2()*(1. + self._a_3L()*(self._omega_L1()/self._omega(f))**2. + self._a_4L()*(self._omega(f)/self._omega_L2())**2. )
		elif [~second_condition] :
			return self._L_inf() + np.sqrt(cst.mu_0/(2.*self._omega(f)*self._kappa))*((self._F_lc() + self._F_lg())/(4.*self._F_up(self._t/2.)**2.))*(1. + self._a_5L()*(self._omega_L2()/self._omega(f)))
		
	
	def get_resistance_per_unit_length(self, f):
		'''Return the length resistance of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Length resistance in Ohms per meter
		'''
		
		return self._Rc(f) + self._Rg(f)
	
	def get_capacitance_per_unit_length(self, f):
		'''Return the length capacitance of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Length capacitance in Farrad per meter
		'''
		
		if isinstance(f, np.ndarray) :
			
			return np.array([2.*cst.epsilon_0*(self._F_up(self._t) + self._epsilon_r*self._F_low())]*len(f))
		else:
			return 2.*cst.epsilon_0*(self._F_up(self._t) + self._epsilon_r*self._F_low())
	
	def get_conductance_per_unit_length(self, f):
		'''Return the length conductance of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Length conductance in Siemens per meter
		'''
		
		return 2.*self._omega(f)*cst.epsilon_0*self._epsilon_r*self._tan_delta*self._F_low()
	
	def get_characteristic_impedance(self, f):
		'''Return the absolute value of the characteristic impedance of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Characteristic impedance in Ohms
		'''
		
		return abs(np.sqrt((self.get_resistance_per_unit_length(f) + 1j*self._omega(f)*self.get_inductance_per_unit_length(f))/(self.get_conductance_per_unit_length(f) + 1j*self._omega(f)*self.get_capacitance_per_unit_length(f))))
	
	def get_complex_wave_vector_per_unit_length(self, f):
		'''Return the absolute value of the complex wave vector coefficient of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Absolute value of the complex wave vector coefficient
		'''
		
		return abs(np.sqrt((self.get_resistance_per_unit_length(f) + 1j*self._omega(f)*self.get_inductance_per_unit_length(f))*(self.get_conductance_per_unit_length(f) + 1j*self._omega(f)*self.get_capacitance_per_unit_length(f))))
	
	def get_attenuation_per_unit_length(self, f):
		'''Return the attenuation coefficient of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Attenuation coefficient
		'''
		return ((self.get_resistance_per_unit_length(f) + 1j*self._omega(f)*self.get_inductance_per_unit_length(f))*(self.get_conductance_per_unit_length(f) + 1j*self._omega(f)*self.get_capacitance_per_unit_length(f))).real
	
	def get_wave_vector_per_unit_length(self, f):
		'''Return the wave vector coefficient of the transmision line
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Beta coefficient
		'''
		return ((self.get_resistance_per_unit_length(f) + 1j*self._omega(f)*self.get_inductance_per_unit_length(f))*(self.get_conductance_per_unit_length(f) + 1j*self._omega(f)*self.get_capacitance_per_unit_length(f))).imag
	
	def get_velocity(self, f):
		'''Return the velocity of the wave in the coplanar wave guide
				- Input :
					- Frequency (float | list | numpy.ndarray) in Hertz
				
				- Output :
					- Velocity in unit of c (speed of light)
		'''
		
		return 1./np.sqrt(self.get_capacitance_per_unit_length(f) * self.get_inductance_per_unit_length(f))/cst.c
	
	
		
	#################################################################################
	#
	#
	#									Optimize
	#
	#
	#################################################################################
	
	def print_parameters(self):
		'''
			Summarize all parameters of the CPW object.
		'''
		
		print '------------------------------------------------------------'
		print '			Parameters'
		print ''
		print '	Central line width:		'+str(self._w)+'		m'
		print '	Gap separation width:		'+str(self._s)+'	m'
		print '	Thickness:			'+str(self._t)+'		m'
		print '	Ground plane width:		'+str(self._w_g)+'		m'
		print ''
		print '	Relative permitivity:		'+str(self._epsilon_r)+'		F/m'
		print '	Loss tangente:			'+str(self._tan_delta)
		print '	Electrical conductivity:	'+str(self._kappa)+'	S/m'
		
	#################################################################################
	#
	#
	#									Optimize
	#
	#
	#################################################################################
	
	def _residual_optimal_gap_separation(self, gapWidth, targetImpedance, targetFrequency, verbose):
		
		if verbose :
			
			print 'Gap width:	'+str(abs(gapWidth[0]))+'	m'
		
		self.set_width_gap_separation(abs(gapWidth))
		
		return self.get_characteristic_impedance(targetFrequency) - targetImpedance
	
	def find_optimal_gap_separation(self, targetImpedance, targetFrequency, verbose=False):
		'''
			Calculate the optimal gap width in order to get a choosen impedance.
			
			Input:
				- targetImpedance (float): Impedance choosen for the optimization in ohm.
				- targetFrequency (float): Frequency at which the optimization will be calculated in Hz. Usually not important.
				- verbose (booleen) : Allows the display of the gap width during the optimization
			
			Output:
				- (finalImpedance, finalGapWidth): reach impedance in ohm, optimal gap width in meter.
		'''
		
		save = self._s
		test = fsolve(self._residual_optimal_gap_separation, self._s ,args=(float(targetImpedance), float(targetFrequency), verbose))
		
		#If the result of the test is true, the optimization failed
		if save == test[0]:
			
			#We set attributs like before
			self._s = save
			self._b   = self._w/2. + save
			
			raise Exception('The optimization failed for the following values: targetImpedance = '+str(targetImpedance)+' ohms and targetFrequency = '+str(targetFrequency)+' hertz.')
		
		else:
			
			return self.get_characteristic_impedance(targetFrequency), self._s
	
	
	
#	def L_eq_lambda4(self, f, l):
#		
#		return 8.*l*self.L_l(f)/cst.pi**2.
#	
#	def C_eq_lambda4(self, f, l):
#		
#		return l*self.C_l(f)/2.
#	
#	def R_eq_lambda4(self, f, l):
#		
#		return self.Z_0(f)/self.alpha(f)/l
#	
#	def Q_lambda4(self, f, l):
#		
#		return cst.pi/(4.*self.alpha(f)*l)
#	
#	def resonance_frequency_lambda4(self, l):
#		f = 9e9
#		return 1./(2.*cst.pi*np.sqrt(self.L_eq_lambda4(f, l) * self.C_eq_lambda4(f, l)))
