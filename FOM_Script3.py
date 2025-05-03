#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:41:30 2024

@author: erik
"""
import numpy as np 
import pandas as pd
from scipy import constants
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import  EngFormatter
from math import sqrt
import math
from scipy.interpolate import interp1d, RegularGridInterpolator
from IPython.display import display, Latex

def myDis(s):
    display(Latex(s))
    return 

def disp_ans(ans):
    #might do the same thing as latex(sym.evalf(3))
   try:
       ans1, ans2 = power_of_10(ans)
       return f'{ans1:.2f}' +  r'\cdot'+ r'10^{'+ f'{ans2}' + r'}\,' 
   except ValueError as e:
       return ans
def power_of_10(num):
    # Get the power of 10
    power = int(math.log10(abs(num)))
    # Adjust the number accordingly
    adjusted_num = num / (10 ** power)
    return adjusted_num, power
   
def process_res(res, unit):
    '''
    

    Parameters
    ----------
    var : variable instance, subclass of sympy Symbol 
    res : result expression from nonlinsolve known to not have free symbols

    Returns
    -------
    A latex string with a number, a (possiblebly greek) letter representing a power of 10
    that is a multiple of 3 (Giga, Mega, kilo, milli, micro, nano) and a unit 
    
    >>>process_res(Lres, 0.000120)
    120 uH

    >>>process_res(fsw, 1234.56)
    1.23 kHz
    '''
    formatter = EngFormatter(places=3)

    expr = formatter.format_eng(res)
    letter = expr[-1]
    num = expr[:-1]
    return  num + r'\,\, \text{'+ letter + unit + r'}'
eps_0 = constants.epsilon_0
rho = 6150 #mass density [kg / m^3]
# Define the stiffness matrix C, values from Popa [Pa = N/m^2]
C = np.array([
    [390, 145, 106, 0, 0, 0],
    [145, 390, 106, 0, 0, 0],
    [106, 106, 398, 0, 0, 0],
    [0, 0, 0, 105, 0, 0],
    [0, 0, 0, 0, 105, 0],
    [0, 0, 0, 0, 0, 122.5]
])*1e9
# Invert the matrix to get the compliance matrix
S = np.linalg.inv(C) #compliance matrix, [m^2/N]

#dielectric constant matrix 
eps = np.array([[9.5,0,0],[0,9.5,0],[0,0,10.4]])*eps_0

#piezo electric strain modulus C/m^2 from Popa 
e = np.array([[ 0.  ,  0.  ,  0.  ,  0.  , -0.3 ,  0.  ],
       [ 0.  ,  0.  ,  0.  , -0.3 ,  0.  ,  0.  ],
       [-0.33, -0.33,  0.65,  0.  ,  0.  ,  0.  ]])
d = e@S


#Reproduce Figure 3-4
fig3_4 = 'Fig3-4.csv'
# Load the CSV file into a pandas DataFrame
df = pd.read_csv(fig3_4)
curve_names = ['S0_no_elec', 'S0_bot_elec', 'A0_no_elec', 'A0_bot_elec']
x_values = df['x'] 
# Create interpolation functions for each curve
ksq_functions = {}
for curve in curve_names:
    ksq_functions[curve] = interp1d(x_values, df[curve], kind='cubic', fill_value="extrapolate")

# Create the figure and axes
fig, ax = plt.subplots()

# Define line styles and colors for each curve
line_styles = {
    'S0_no_elec': {'linestyle': '--', 'color': 'black', 'label': r'$S_0$, no bottom electrode'},
    'S0_bot_elec': {'linestyle': '-', 'color': 'black', 'label': r'$S_0$, bottom electrode'},
    'A0_no_elec': {'linestyle': '-.', 'color': 'red', 'label': r'$A_0$, no bottom electrode'},
    'A0_bot_elec': {'linestyle': '-', 'color': 'red', 'label': r'$A_0$, bottom electrode'},
}

# Plot each curve with the specified styles using the ax object
for curve in curve_names:
    ax.plot(x_values, df[curve], 
            linestyle=line_styles[curve]['linestyle'], 
            color=line_styles[curve]['color'], 
            label=line_styles[curve]['label'])

# Add a horizontal line for Rayleigh
ax.axhline(y=0.1, color='blue', linestyle=':', label='Rayleigh')
ax.set_ylim(0,.8)
ax.set_xlim(0,1)
# Add labels and title
ax.set_xlabel(r'Normalized GaN thickness, $h_{\mathrm{GaN}}/\lambda$', fontsize=12)
ax.set_ylabel(r'Electromechanical coupling, $k^2$ (%)', fontsize=12)
ax.set_title('Electromechanical Coupling Curves', fontsize=14)

# Add a legend
ax.legend(loc='best', fontsize=10)
plt.show()
###################
###################
# fQ interpolation 
df = pd.read_csv('fQ-vs-f.csv')#, index_col='x')
start = [1e15, 1e7]  # Start values for x and y axes
stop = [1e17, 2e10]  # Stop values for x and y axes

# Create a 2D logarithmic spaced array
x = np.logspace(np.log10(start[0]), np.log10(stop[0]), 3)  # x-axis
y = np.logspace(np.log10(start[1]), np.log10(stop[1]), 44)  # y-axis

data = df[['Nd10^15','Nd10^16','Nd10^17']]
data_arr = np.array(data)
l = data_arr.shape[0]
interp = RegularGridInterpolator((y,x), data_arr,
                                 bounds_error=False, fill_value=None)

f = np.logspace(5, 10.3,44)
fQ15 = interp((f,np.ones(l)*1e15))
fQ16 = interp((f,np.ones(l)*1e16))
fQ17 = interp((f,np.ones(l)*1e17))
fig, ax = plt.subplots()
# Define a 3-color cycle
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
color_cycler = cycler('color', colors)
ax.set_prop_cycle(color_cycler)
for c in df.columns.drop(['x', 'ank-lr']):
    ax.plot(df['x'], df[c], label=c)
ax.plot(f, fQ15, label=r'$\text{interp}\, 10^{15}$', linestyle='none', marker='*',markevery=3)
ax.plot(f, fQ16, label='interp 16', linestyle='none', marker='*',markevery=3)
ax.plot(f, fQ17, label='interp 17', linestyle='none', marker='*',markevery=3)
#ax.plot(f, fQ_interp15)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e10,1e14)
ax.set_xlim(1e5,2e10)
ax.set_title('fQ vs f from Fig 3.10')
ax.set_ylabel('fQ product ')
ax.set_xlabel('Frequency [Hz]')
ax.legend()
plt.show()
fig, ax = plt.subplots()
#for c in df.columns.drop('x'):
#    ax.plot(df['x'], df[c], label=c)
ax.plot(f, fQ15/f, label=r'$\text{interpolated}\, N_d = 10^{15}$')
ax.plot(f, fQ16/f, label=r'$\text{interpolated}\, N_d = 10^{16}$')
ax.plot(f, fQ17/f, label=r'$\text{interpolated}\, N_d = 10^{17}$')
#ax.plot((freq), (interp((freq, 6e16))/freq), linestyle='None', marker='*')

ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_ylim(1e10,1e14)
ax.set_xlim(1e5,2e10)
ax.legend()
ax.set_title('Q vs f for GaN')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Q')
plt.show()




print('Material Constants\n')
s11 = S[0,0]
d31 = d[2,0]

myDis('$s_{11} = ' + disp_ans(S[0,0]) +\
      r' \frac{\text{m}^2}{\text{N}}$')
    

rho_str = r'$\rho =  '+ f'{rho}' + r'  \frac{\text{kg}}{\text{m}^3}$'
myDis(rho_str)

d_str = r'$d_{31} ='+ disp_ans(d31) +  r'\frac{\text{C}}{\text{N}}$'
myDis(d_str)
va = 1/sqrt(rho*s11)
va_str = r'$v_a = \frac{1}{\sqrt{\rho s_{11}^E}}= '+\
    f'{va:.0f}' + r'\, \frac{\text{m}}{\text{s}}$'
myDis(va_str)

e33 = eps[2,2]
eps = r'$\epsilon_{33}^T = \epsilon_{r,33}\cdot \epsilon_0=' +\
      disp_ans(e33) + r'=\frac{\text{F}}{\text{m}}$'   
myDis(eps)

k31 = abs(d31)/sqrt(s11*e33)
k_str = r'$k_{31} = \frac{|d_{31}|}{\sqrt{s_{11}^E \, \epsilon_{33}^T}}=' +\
    f'{k31:.3f}$'
myDis(k_str)
k31_sq = k31**2
ksq_str = r'$k_{31}^2 = '+f'{k31_sq*100:.3f}'+r'\%$'

g = sqrt(np.pi**2+8*k31**2/(1-k31**2))
g_str = r'$\gamma_0 = \sqrt{\pi^2 + \frac{8k^2}{1 - k^2}}=' + f'{g:.2f}$'
myDis(g_str)
'''
Options for Figure 3-4 extracted k²:
    
r'S0, no bottom electrode'
r'S0, bottom electrode'
r'A0, no bottom electrode'
r'A0, bottom electrode')
'''
hGaN = 2.5e-6 #thickness [m]
lmda = 4e-5 # wavelength [m]
b = 4e-6 # width dimension  [m]
x = hGaN/lmda

option = r'S0, no bottom electrode'
if option == r'S0, no bottom electrode':
    ksq_perc = ksq_functions['S0_no_elec'](x).item()
elif option == r'S0, bottom electrode':
    ksq_perc = ksq_functions['S0_bot_elec'](x).item()
elif option == r'A0, no bottom electrode':
    ksq_perc = ksq_functions['A0_no_elec'](x).item()
elif option == r'A0, bottom electrode': 
    ksq_perc = ksq_functions['A0_bot_elec'](x).item()
    

a = lmda * .75
kappa = 2*np.pi / lmda
Gf = 1/a
kappa_0 = kappa / Gf 
A = 2*b*2*a
Cpth = e33*(1-k31**2)*A/hGaN
Cpemp = e33*(1-ksq_perc/100)*A/hGaN
Csth = 8*k31**2*Cpth/(np.pi**2*(1-k31**2))
Csemp = 8*(ksq_perc/100)*Cpemp/(np.pi**2*(1-ksq_perc/100))
Lsth = (1-k31**2)/(2*Gf**2*k31**2*va**2*Cpth)
Lsemp = (1-(ksq_perc/100))/(2*Gf**2*(ksq_perc/100)*va**2*Cpemp)




myDis(r'$h_{GaN} = ' +disp_ans(hGaN) + r'\text{m}=' + f'{hGaN/1e-6:.2f}' + r'\mu\text{m}$' )

print('\nMaterial k², theoretical:')
myDis(ksq_str)
print('\nEmpirical k² = f(hGaN /λ) = ')
myDis(r'$k^2 = ' +f'{ksq_perc:.3f}'+'\%$')

myDis(r'$a = \frac{3}{4}\lambda='+disp_ans(a) + r'\text{m}=' + f'{a/1e-6:.2f}' + r'\mu\text{m}$' )

myDis(r'$b = ' +disp_ans(b) + r'\text{m}=' + f'{b/1e-6:.2f}' + r'\mu\text{m}$' )
myDis(r'$G_f = \frac{1}{a} ='+ disp_ans(1/a) + r'\,[\text{m}^{-1}]$' )
myDis(r'$\kappa_0 = \frac{\kappa}{G_f} = \kappa a =' +\
      f'{kappa_0:.2f}'+ r'\,[\text{m}^{-1}]$')

myDis(r'$\kappa = \frac{2\pi}{\lambda} = ' +\
      disp_ans(kappa) + r'\,[\text{m}^{-1}]$')
 
myDis(r'$\text{area}\, A = ' + disp_ans(A*(1e6)**2)+ r'\mu\text{m}^2$')

myDis(r'$C_p = \epsilon^T (1 - k_{emp}^2) \frac{A}{2l}' + \
      disp_ans())
myDis(r'$C_p = \epsilon^T (1 - k_{th}^2) \frac{A}{2l}')
myDis(r'$L_s = \frac{1 - k_{emp}^2}{2G_f^2 k_{emp}^2 v_a^2 C_{p+}}$')

myDis(r'$L_s = \frac{1 - k_{th}^2}{2G_f^2 k_{th}^2 v_a^2 C_{p+}}$')











