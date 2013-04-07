CPW
===

CPW is a class allowing the calculation of Coplanar Waveguide characteristic values following geometric parameters, conductivity, relative permitivity and, loss tangent.

All method are well documented into the class.


Example:

`````python

import CPW as CPW

line = CPW.CPW()

line.set_thickness(0.5e-6)

line.get_inductance_per_unit_length(2e9)
//4.26e-7

`````


This class is based on an article of Wolfgang Hainrich
"Quasi-TEM Description of MMIC coplanar Lines Including onductor-Loss Effects"
 IEEE Transactions on Microwave Theory And Techniques, vol 41, n° 1, January 1993

