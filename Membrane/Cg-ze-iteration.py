#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:00:28 2024

@author: erik
"""
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, EngFormatter
graphene={'Ey': 1e12 ,  'v':.19 }
# Constants
epsilon_0 = 8.854187817e-12  # Permittivity of free space in F/m
E_Y = graphene['Ey']  # Example Young's modulus in Pa
nu = graphene['v']  # Poisson's ratio
t = 10e-9  # Thickness in meters
R = 3e-6  # Radius in meters
V_g = 1.0  # Voltage in Volts
z_0 = 150e-9  # Example gap distance in meters
epsilon_0r  = .0001 #initial strain

def Cgc_integrand(r, R, ze, zo):
    return (epsilon_0 * r) / (zo - ze * (1 / R**2) * (R**2 - r**2))

def compute_Cgc(R, ze, zo):
    # Define the limits of integration
    theta_min, theta_max = 0, 2 * np.pi
    r_min, r_max = R*1e-9, R*.999

    # Perform the double integration
    result, error = quad(Cgc_integrand,r_min, r_max, args=(R, ze, zo))
    #result, error = dblquad(Cgc_integrand, theta_min, theta_max, r_min, r_max, args=(R, ze, zo))
    return result*2*np.pi


# Define the integrand for Cg'
def integrand(r, R, ze, zo):
    numerator = r * (R**2 - r**2) / R**2
    denominator = (zo - ze * (R**2 - r**2) / R**2)**2
    return numerator / denominator

# Compute the derivative of the capacitance
def Cg_prime(ze, R, zo):
    result, error = quad(integrand, R*1e-9, R*.999, args=(R, ze, zo))
    return 2 * np.pi * epsilon_0 * result

# Define the nonlinear equation
def equation(ze):
    term1 = (8 * np.pi * E_Y * t * ze**3) / (3 * (1 - nu**2) * R**2)
    term2 = 2 * np.pi * E_Y * t * epsilon_0r * ze
    term3 = 0.5 * Cg_prime(ze, R, z_0) * V_g**2
    return term1 + term2 - term3

ze = np.logspace(-10, np.log10(z_0))
term1 = (8 * np.pi * E_Y * t * ze**3) / (3 * (1 - nu**2) * R**2)
term2 = 2 * np.pi * E_Y * t * epsilon_0r * ze
term3 = 0.5 * np.vectorize(Cg_prime)(ze, R, z_0) * V_g**2
fig, ax = plt.subplots()
ax.plot(ze*1e9, term1, label ='term1')
ax.plot(ze*1e9, term2, label='term2')
ax.plot(ze*1e9, term3, label='term3')
ax.set_xlabel('displacement from eq pos [nm]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.set_title('eq 14, need to find roots')

'''
# Solve for ze using fsolve
ze_initial_guess = 10e-9  # Initial guess for ze = 10nm 
ze_solution = fsolve(equation, ze_initial_guess)

print(f"Solution for ze: {ze_solution[0]:.6e} meters")

Cgc = compute_Cgc(R, ze_solution[0], z_0)
#Cgc = compute_Cgc(R, 10e-9, 150e-9)
print(f"Cgc: {Cgc} F")

ze_list = np.linspace(2e-9, .3*z_0)
eq2 = np.vectorize(equation)

fze = eq2(ze_list)
fig, ax = plt.subplots()
ax.plot(ze_list, fze)
ax.plot([ze_solution, ze_solution], [-1, 1], color='r')
ax.set_ylim(-.0003, .0003)
'''