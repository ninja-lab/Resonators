#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 09:55:34 2025

@author: erik
"""

import os 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import matplotlib.patches as patches
from scipy.optimize import least_squares

filename = 'S0_lambda10.xlsx'
df = pd.read_excel(filename)
df.columns



# measured or simulated frequency points (Hz) and complex Y (S)
f = df['freq (Hz)']        # 1-D array
df['Admittance (S)'] = df['Admittance (S)'].str.replace('i', 'j')

# Now convert to complex type
Ydata = df['Admittance (S)'].astype('complex128')
fs = 625094243 #
C0_guess = 8.85e-11
k2_guess = .006
Cm_guess = k2_guess*C0_guess
#Cm_guess = C0_guess/100
Lm_guess = 1/((2*np.pi*fs)**2*Cm_guess)
Q = 3000
Rm_guess = np.sqrt(Lm_guess/Cm_guess)/Q
w = 2*np.pi*f
scale = 1e9 

def Y_bvd(p, w):
    C0, Cm, Lm, Rm = p
    Zm = Rm + 1j*(w*Lm - 1/(w*Cm))
    return 1j*w*C0 + 1/Zm
    
def Y_bvd_scaled(p, w):
    return scale * Y_bvd(p, w)  # compare scaled model to scaled data
    
def residual(p, w, Y):
    return np.hstack((np.real(Y_bvd(p, w) - Y),
                      np.imag(Y_bvd(p, w) - Y)))

p0 = [C0_guess, Cm_guess, Lm_guess, Rm_guess]
bounds = (
    [1e-14, 1e-15, 1e-11, 1e-3],  # lower bounds
    [1e-10, 1e-10, 1e-5, 1e+1],   # upper bounds
)

#sol = least_squares(residual, p0, args=(w, Ydata))
sol = least_squares(
    residual,
    p0,
    args=(w, Ydata),
    #method='trf',
    #xtol=1e-15,
    #ftol=1e-15,
    #gtol=1e-15,
    max_nfev=1000,
    #bounds=bounds
)

C0, Cm, Lm, Rm = sol.x
k2_eff = Cm / (C0 + Cm)
#Qm     = 1/(Rm * w[np.argmin(np.imag(Ydata))] * Cm)
Qm     = np.sqrt(Lm / Cm)/Rm
print(f'k_eff^2 = {k2_eff:.4f},   Q_m = {Qm:.0f}')
print('Guess:', p0)
print('Solution:', sol.x)
print('Delta:', sol.x - p0)

fp = np.sqrt(fs**2 / (1 - k2_guess))
fstp = (fp - fs) / 333
f_dense = np.linspace(fs - 333*fstp, fp + 333*fstp, 1000)
w_dense = 2 * np.pi * f_dense
Yfit = Y_bvd(sol.x, w_dense)
Yguess = Y_bvd(p0, w_dense)

plt.figure(figsize=(10, 5))
plt.plot(f, 20*np.log10(np.abs(Ydata)), label='Measured')
plt.plot(f_dense, 20*np.log10(np.abs(Yfit)), label='Fitted', linestyle='--')
plt.plot(f_dense, 20*np.log10(np.abs(Yguess)), label='Guess', linestyle='none', marker= '*', markevery=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Admittance (dB)')
plt.legend()
plt.grid(True)
plt.title('BVD Model Fit vs Measured Admittance ')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(f, np.real(Ydata), label='Measured')
plt.plot(f_dense, np.real(Yfit), label='Fitted', linestyle='--')
plt.plot(f_dense, np.real(Yguess), label='Guess', linestyle='none', marker= '*', markevery=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('real(admittance)')
plt.legend()
plt.grid(True)
plt.title('BVD Model Fit vs Measured Admittance')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(f, np.imag(Ydata), label='Measured')
plt.plot(f_dense, np.imag(Yfit), label='Fitted', linestyle='--')
plt.plot(f_dense, np.imag(Yguess), label='Guess', linestyle='none', marker= '*', markevery=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('imag(admittance)')
plt.legend()
plt.grid(True)
plt.title('BVD Model Fit vs Measured Admittance')
plt.show()