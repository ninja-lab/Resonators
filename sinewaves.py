#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:17:27 2024

@author: erik
"""
import numpy as np
import matplotlib.pyplot as plt

# Define time range
t = np.linspace(0, 1, 1000)

# Define two frequencies, one 10x the other
freq1 = 1  # 1 Hz
freq2 = 5  # 10 Hz
freq3 = 3.3 

# Generate sine waves
wave1 = np.sin(2 * np.pi * freq1 * t)
wave2 = np.sin(2 * np.pi * freq2 * t)

wave3 = [.75 if el > 0 else -.75 for el in np.sin(2 * np.pi * freq3 * t - 5*np.pi/6) ]

# Create plot

fig, ax = plt.subplots() 
ax.plot(t, wave1, 'k', label='1 Hz Sine Wave')
ax.plot(t, wave2,'r', label='10 Hz Sine Wave')
ax.plot(t, wave3,'b', label='10 Hz Sine Wave')
# Add labels and legend
ax.set_title('Sine Waves with Frequencies 1 Hz and 10 Hz')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
#plt.legend()
#plt.grid(True)

# Show plot
plt.show()

fig, ax = plt.subplots() 
ax.plot(t, np.sin(2 * np.pi * freq1*1.3 * t), 'k')
ax.plot(t, np.sin(2 * np.pi * freq1*1.3 * t - 2*np.pi/3), 'k')
ax.plot(t, np.sin(2 * np.pi * freq1*1.3 * t + 2*np.pi/3), 'k')
plt.show()
