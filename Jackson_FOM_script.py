#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:32:39 2024

@author: erik
"""

# based on Laura Popa and Jessica Boles theses,
import numpy as np
from math import sqrt 
hgan = 4E-6  # m
hgan_lmda = 0.1
fQ = 2E13
k2 = 0.008 #from highest part of bottom electrode curve in fig 3.4(c)
k2 = .0035 #from Figure 3.4(c) maximum at hgan/lambda = .45
va = 7750  # m/s from
#Gallium Nitride as an Electromechanical Material reports acoustic
#velocity to be 8044 m/s 
#could calculate acoustic velocity from Boles table 6.3
# va= 1/(sqrt(rho*sE11))
#rho = 6150 kg / m^3 (appendix in Popa matches GaN as EM Matl)
#sE11 is component of compliance tensor but I have stiffness tensor components
print(1/sqrt(6150*3.1*1e-12)) 
#1/sqrt(kg / m^3 * m^2/N)
#kg/N = kg  m^2/s^2 / kg
print(1/sqrt(6150*4.1*1e-12)) #from 
lmda = hgan/hgan_lmda
f = va/lmda
Q = fQ/f
gamma0 = np.sqrt(np.pi**2+8*(k2/(1-k2)))
k0 = np.pi*gamma0/(np.pi+gamma0)

fom_m = 4*Q*k2/(1-k2)*1/(np.pi**2*k0)  # Pout/Ploss

# power density metrics
# Smax =   # yield strain
# Tmax =   # yield stress
# Emax =   # lower of breakdown or coercive field
# As =  # effective area of cooling for resonator
# H = 1 # Ploss per area As
#
# ILSmaxo = va*d31/s11*Smax*np.sin(k0)
# ILTmaxo = va*d31*Tmax*np.cot(k0/2)
# ILEmaxo = va*k2*eps33*Emax*tan(k0)
# ILHmaxo = sqrt(4*Q*k2*eps33*va/np.pi*H)