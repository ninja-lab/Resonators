#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:26:16 2024

@author: erik
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
m = 1.0      # mass
c = 0.01      # damping coefficient
k = 1      # linear stiffness
alpha = 0.05  # Duffing nonlinearity coefficient
F0 = 1.0     # amplitude of external force

# Time range for the simulation
t_start = 0.0
t_end = 100.0
t_eval = np.linspace(t_start, t_end, 10000)

# Function to model the SDOF system with Duffing nonlinearity
def duffing_oscillator(t, y, omega):
    x, v = y
    dxdt = v
    dvdt = (F0 * np.cos(omega * t) - c * v - k * x - alpha * x**3) / m
    return [dxdt, dvdt]

# Function to solve the system for a given frequency
def solve_for_frequency(omega):
    # Initial conditions: [displacement, velocity]
    y0 = [0.0, 0.0]
    sol = solve_ivp(duffing_oscillator, [t_start, t_end], y0, args=(omega,), t_eval=t_eval, method='RK45')
    return sol

# Frequencies to evaluate
frequencies = np.linspace(0.1, 2.0, 50)

# Calculate the amplitude of the response for each frequency
amplitudes = []
for omega in frequencies:
    sol = solve_for_frequency(omega)
    x = sol.y[0]
    amplitude = np.max(np.abs(x))
    amplitudes.append(amplitude)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(frequencies, amplitudes, label='Displacement Amplitude')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Amplitude')
plt.title('Displacement Amplitude vs Frequency for Duffing Oscillator')
plt.legend()
plt.grid(True)
plt.show()
