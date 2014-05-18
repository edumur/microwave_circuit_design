CPW
===

CPW is a class allowing the calculation of Coplanar Waveguide characteristic values following geometric parameters, conductivity, relative permitivity and, loss tangent.

All method are well documented into the class.


Example:

`````python

import CPW

line = CPW.CPW()

//You can change your cpw parameters thanks to set method
line.set_thickness(0.5e-6)
line.set_width_central_line(10e-6)

//You can easily calculate physical parameters
line.get_inductance_per_unit_length(2e9)
>>>>4.26e-7
line.get_velocity(2e9)
>>>>0.399

//You can optimized your chracteristic impedance
a.find_optimal_gap_separation(50., 10e9)
>>>>(49.9999, 5.38152e-6)

`````


This class is based on an article of Wolfgang Hainrich
"Quasi-TEM Description of MMIC coplanar Lines Including onductor-Loss Effects"
 IEEE Transactions on Microwave Theory And Techniques, vol 41, n° 1, January 1993

