#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:15:09 2025

@author: erik
"""
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


fs = 625094243 #
C0_guess = 8.85e-11
k2_guess = .006
Cm_guess = k2_guess*C0_guess
#Cm_guess = C0_guess/100
Lm_guess = 1/((2*np.pi*fs)**2*Cm_guess)
Q = 3000
Rm_guess = np.sqrt(Lm_guess/Cm_guess)/Q
w      = 2*np.pi*df['freq (Hz)'].values
Ydata  = df['Admittance (S)'].astype('complex128').values
# 1. choose *scales* that bring every parameter close to 1
scales  = np.array([1e-11, 1e-13, 1e-7, 1])   # [C0, Cm, Lm, Rm]

p0      = np.array([C0_guess, Cm_guess, Lm_guess, Rm_guess])
p0_hat  = p0 / scales          # scaled initial guess (all ~ O(1))

def Y_bvd(p_hat, w):
    """BVD admittance using *unscaled* parameters."""
    C0, Cm, Lm, Rm = p_hat*scales
    Zm = Rm + 1j*(w*Lm - 1/(w*Cm))
    return 1j*w*C0 + 1/Zm

def residual(p_hat, w, Y):
    res = Y_bvd(p_hat, w) - Y
    return np.concatenate((res.real, res.imag))

def jacobian(p_hat, w, Y):
    C0, Cm, Lm, Rm = p_hat * scales
    Zm       = Rm + 1j*(w*Lm - 1/(w*Cm))
    inv_Zm   = 1/Zm
    inv_Zm2  = inv_Zm**2

    # analytic grads (un‑scaled)
    dC0 = 1j*w
    dRm = -inv_Zm2
    dLm = -1j*w * inv_Zm2
    dCm = -1j/(w*Cm**2) * inv_Zm2

    # stack and apply scaling factors
    grads = np.vstack((dC0, dCm, dLm, dRm)).T * scales   # shape (N,4)

    # split into real/imag blocks expected by least_squares
    J = np.vstack((grads.real, grads.imag))              # (2N,4)
    return J

sol = least_squares(
        residual, p0_hat, args=(w, Ydata),
        x_scale='jac',         # use Jacobian columns to scale the trust region
        diff_step=1e-3,        # absolute step in *scaled* space (1e‑3 ≈ 0.1 %)
        max_nfev=2000,
        jac = jacobian
      )
p_opt = sol.x * scales 
C0, Cm, Lm, Rm = p_opt
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
Yfit = Y_bvd(p_opt/scales, w_dense)
Yguess = Y_bvd(p0/scales, w_dense)

plt.figure(figsize=(10, 5))
plt.plot(f, 20*np.log10(np.abs(Ydata)), label='Measured')
plt.plot(f_dense, 20*np.log10(np.abs(Yfit)), label='Fitted', linestyle='--')
plt.plot(f_dense, 20*np.log10(np.abs(Yguess)), label='Guess', linestyle='none', marker= '*', markevery=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Admittance (dB)')
plt.legend()
plt.grid(True)
plt.title('BVD Model Fit vs Measured Admittance - Improved')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(f, np.real(Ydata), label='Measured')
plt.plot(f_dense, np.real(Yfit), label='Fitted', linestyle='--')
plt.plot(f_dense, np.real(Yguess), label='Guess', linestyle='none', marker= '*', markevery=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('real(admittance)')
plt.legend()
plt.grid(True)
plt.title('BVD Model Fit vs Measured Admittance - Improved')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(f, np.imag(Ydata), label='Measured')
plt.plot(f_dense, np.imag(Yfit), label='Fitted', linestyle='--')
plt.plot(f_dense, np.imag(Yguess), label='Guess', linestyle='none', marker= '*', markevery=20)
plt.xlabel('Frequency (Hz)')
plt.ylabel('imag(admittance)')
plt.legend()
plt.grid(True)
plt.title('BVD Model Fit vs Measured Admittance - Improved')
plt.show()