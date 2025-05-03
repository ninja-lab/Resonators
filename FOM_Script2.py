#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:17:10 2024

@author: erik
"""

import streamlit as st 
import numpy as np 
import pandas as pd
import sympy
from scipy import constants
from sympy import Matrix, print_latex, Float 
from IPython.display import display
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import  EngFormatter, MaxNLocator, LogLocator
#import math
from math import sqrt
from sympy import symbols, pi, sqrt, solve, nonlinsolve, Rational, nsolve
from sympy.printing import latex
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from FOM_Script_Helpers import *
from lcapy import R, L, C, Circuit, j2pif 
eps_0 = constants.epsilon_0

# Define the stiffness matrix C, values from Popa [Pa = N/m^2]
Cmat = np.array([
    [390, 145, 106, 0, 0, 0],
    [145, 390, 106, 0, 0, 0],
    [106, 106, 398, 0, 0, 0],
    [0, 0, 0, 105, 0, 0],
    [0, 0, 0, 0, 105, 0],
    [0, 0, 0, 0, 0, 122.5]
])*1e9
# Invert the matrix to get the compliance matrix
Smat = np.linalg.inv(Cmat) #compliance matrix, [m^2/N]

#dielectric constant matrix 
eps = np.array([[9.5,0,0],[0,9.5,0],[0,0,10.4]])*eps_0

#piezo electric strain modulus C/m^2 from Popa 
e = np.array([[ 0.  ,  0.  ,  0.  ,  0.  , -0.3 ,  0.  ],
       [ 0.  ,  0.  ,  0.  , -0.3 ,  0.  ,  0.  ],
       [-0.33, -0.33,  0.65,  0.  ,  0.  ,  0.  ]])
e_mat = Matrix(e)
s_mat = Matrix(Smat)
#piezoelectric charge constant matrix C/N
d = e_mat*s_mat
rho = 6150 #mass density [kg / m^3]

fig3_4 = 'Fig3-4.csv'
# Load the CSV file into a pandas DataFrame
df = pd.read_csv(fig3_4)
curve_names = ['S0_no_elec', 'S0_bot_elec', 'A0_no_elec', 'A0_bot_elec']
x_values = df['x'] 
# Create interpolation functions for each curve
ksq_functions = {}
for curve in curve_names:
    ksq_functions[curve] = interp1d(x_values, df[curve], kind='cubic', fill_value="extrapolate")

if 'x' not in st.session_state:
    st.session_state['x'] = .5

df = pd.read_csv('fQ-vs-f.csv')
data = df[['Nd10^15','Nd10^16','Nd10^17']]
start = [1e15, 1e7]  # Start values for x and y axes
stop = [1e17, 2e10]  # Stop values for x and y axes
conc = np.logspace(np.log10(start[0]), np.log10(stop[0]), 3)  
freqs = np.logspace(np.log10(start[1]), np.log10(stop[1]), 44)
fQ = RegularGridInterpolator((freqs, conc), np.array(data),
                     bounds_error=False, fill_value=None)


constants, geometry, calcs = st.tabs(["Material Constants", "Geometry", "Calculations"])
with constants:
    with st.expander('Material Constants for __ GaN'):
        st.title('Material Constants for __ GaN')
    
        C_scientific = Matrix(Cmat).applyfunc(formatter)
        st.header(r'Stiffness Matrix $\mathbf{C} \,\frac{\text{N}}{\text{m}^2}$')
        st.latex(latex(C_scientific)) #m^2 / N   
        
        S_scientific = Matrix(Smat).applyfunc(lambda x: '{:.2e}'.format(float(x)))
        st.header(r'Compliance Matrix $\mathbf{S}=\mathbf{C}^{-1}\,\,'+\
                  r' \frac{\text{m}^2}{\text{N}}$')
        st.latex(latex(S_scientific)) #m^2 / N    
        
        st.header(r'Piezoelectric Strain Modulus $\mathbf{e} \,\frac{\text{C}}{\text{m}^2}$ ')
        st.latex(latex(e_mat))
        
        st.header(r'Piezoelectric Charge Constant $\mathbf{d}=\mathbf{e}\mathbf{S}'+\
                  r' \,\frac{\text{C}}{\text{N}}$ ')
        st.latex(latex(d.applyfunc(lambda x: '{:.2e}'.format(float(x)))))    
        
        st.header(r'Dielectric Constant $\mathbf{\epsilon}\, \frac{\text{F}}{\text{m}}$ ')
        eps_mat = Matrix(eps).applyfunc(lambda x: '{:.2e}'.format(float(x)))
        st.latex(latex(eps_mat))   
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        s11 = float(S_scientific[0,0])
        s11_str = r's_{11}^E =' + disp_ans(s11) +r' \frac{\text{m}^2}{\text{N}} '
        st.latex(s11_str)
        
        rho_str = r'\rho =  '+ f'{rho}' + r'  \frac{\text{kg}}{\text{m}^3}'
        st.latex(rho_str)
        
        d31 = d[2,0]
        d_str = r'd_{31} ='+ disp_ans(d31) +  r'\frac{\text{C}}{\text{N}} '
        st.latex(d_str)
        va = 1/np.sqrt(rho*s11)
        st.session_state['va'] = va
        va_str = r'v_a = \frac{1}{\sqrt{\rho s_{11}^E}}= '+\
            f'{va:.0f}' + r'\, \frac{\text{m}}{\text{s}}'
        st.latex(va_str)
    
    with col2:
        #ans = 8.9*eps_0
        e33 = eps_mat[2,2]
        eps = r'\epsilon_{33}^T = \epsilon_{r,33}\cdot \epsilon_0=' +\
              disp_ans(e33) + r'\frac{\text{F}}{\text{m}}'   
        st.latex(eps)
        k31 = abs(d31)/sqrt(s11*e33)
        k_str = r'k_{31} = \frac{|d_{31}|}{\sqrt{s_{11}^E \, \epsilon_{33}^T}}=' +\
            f'{k31:.3f}'
        st.latex(k_str)
        k31_sq = k31**2
        ksq_str = r'k_{31}^2 = '+f'{k31_sq*100:.3f}'+r'\%'
        st.latex(ksq_str)
        g = sqrt(np.pi**2+8*k31**2/(1-k31**2))
        g_str = r'\gamma_0 = \sqrt{\pi^2 + \frac{8k^2}{1 - k^2}}=' + f'{g:.2f}'
        st.latex(g_str)
        st.session_state['e33'] = e33
        st.session_state['k31_sq'] = k31_sq
        st.session_state['gamma'] = g
    st.divider()
    option = st.selectbox("Configuration?", 
                          (r'S0, no bottom electrode',r'S0, bottom electrode',
                           r'A0, no bottom electrode', r'A0, bottom electrode'),
                          key='electrode_config')
  
    x = st.session_state['x']
    if option == r'S0, no bottom electrode':
        ksq_perc = ksq_functions['S0_no_elec'](x)
    elif option == r'S0, bottom electrode':
        ksq_perc  = ksq_functions['S0_bot_elec'](x)
    elif option == r'A0, no bottom electrode':
        ksq_perc  = ksq_functions['A0_no_elec'](x)
    elif option == r'A0, bottom electrode': 
        ksq_perc  = ksq_functions['A0_bot_elec'](x)
    st.latex(r'k^2=' + f'{ksq_perc :.3f}'+r'\%')
    st.latex(r'x = h_{GaN} / \lambda = '+ f'{st.session_state["x"]:.3f}')
with geometry: 
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            st.image('fig6-1a.png', width=200)
        with cols[1]:
            st.image('JA_dims.png', width=200)
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            
            st.number_input(label=r'$h_{GaN} = 2l\, [\text{m}]$', key='hGaN',min_value=1e-6,
                            max_value=5.0e-6, step=.1e-6,value=2.0e-6, format='%.2e')
            st.number_input(label=r'$\lambda \,[\text{m}]$', key='lambda', min_value=10*hGaN_min,
                            max_value=1000*hGaN_max, step=1e-6,value=20*hGaN_min, format='%.2e')
            st.number_input(label=r'$2b\,[\text{m}]$',min_value=.1e-6,
                            max_value=10.0e-6, step=.1e-6,value=4.0e-6, format='%.2e',key='2b')
            st.session_state['l'] = st.session_state['hGaN']/2
            st.session_state['x'] = st.session_state['hGaN']/st.session_state['lambda']
            st.latex(r'x = '+ f'{st.session_state["x"]:.2f}')
            a = st.session_state["lambda"]*.75
            st.session_state['a'] = a
            st.latex(r'a = '+ f'{a*1e6:.2f}' +  r'\,\text{[μm]}')
            
            st.latex(r'b = '+ f'{st.session_state["2b"]*1e6/2:.2f}' + \
                     r'\,\text{[μm]}')
            st.latex(r'h_{Gan} = 2l = '+ f'{st.session_state["hGaN"]*1e6:.2f}' + \
                     r'\,\text{[μm]}')       
            
            kappa = 2*np.pi/st.session_state['lambda']
            st.session_state['kappa'] = kappa
            st.latex(r'\kappa = ' + disp_ans(kappa)) # f'{kappa:.2e}')
            st.latex(r'G_f = ' + disp_ans(1/a) + r'\,[\text{m}^{-1}]')#f'{1/a:.2e}')
            kappa_0 = kappa * a
            st.session_state['kappa_0'] = kappa_0
            st.latex(r'\kappa_0 ='+f'{kappa_0:.2f}')
            
            A = st.session_state['2b']*2*st.session_state['a']
            st.latex(r'A = '+ disp_ans(A*(1e6)**2) +r'[μm^2]')
            
            Cp = e33*(1-k31**2)*A/st.session_state['hGaN']
            st.latex(r'C_p = ' + disp_ans(Cp) + '[F]')
            
            Cs = 8*k31**2*Cp/(np.pi**2*(1-k31**2))
            st.latex(r'C_s = ' + disp_ans(Cs) + '[F]')     
            Gf = 1/a 
            Ls = (1-k31**2)/(2*Gf**2*k31**2*va**2*Cp)
            st.latex('L_s = ' + disp_ans(Ls) + r'\text{[H]}')
            st.session_state['Cs'] = float(Cs)
            st.session_state['Ls'] = float(Ls)
            st.session_state['Cp'] = float(Cp) 
            st.session_state['A'] = A
            st.session_state['Gf'] = Gf
            
            
        with cols[1]:
            
            
            st.latex(r'f\cdot \lambda = v_a')
            f = st.session_state['va']/st.session_state['lambda']
            f_str = r'f = ' +disp_ans(f) + r'\text{[Hz]}'#f'{f:.2e}'
            st.latex(f_str)
            
            Q = fQ((f, 1e16) )/f
            Q_str = r'Q = ' + disp_ans(Q)
            st.latex(Q_str)
            #st.write(float(L/C))
            #st.write(type(L/C))
            Rs = np.sqrt(st.session_state['Ls']/st.session_state['Cs'] )/Q
            st.session_state['Rs'] = float(Rs)  
            #Rs = 100
            st.latex('R_s = '+ disp_ans(Rs))
            st.latex(r'\frac{h_{GaN}}{\lambda} = x ')
            st.latex(r'2a = \frac{3}{2}\lambda')
            st.latex(r'\kappa = \frac{2\pi}{\lambda}')
            st.latex(r'G_f = \frac{1}{a}')
            st.latex(r'\kappa_0 = \frac{\kappa}{G_f} = \kappa a')
            
            fs = 1/(2*pi*sqrt(st.session_state['Ls']*st.session_state['Cs'] ))
            st.latex(r'f_s = ' + latex(fs.evalf(5)) + r'\,\text{Hz}')
            fp = 1/(2*pi*sqrt(st.session_state['Ls']*st.session_state['Cs']* \
                              (1+st.session_state['Cs']/st.session_state['Cp'])))
            st.latex(r'f_p = ' + latex(fp.evalf(5)) + r'\,\text{Hz}')
            st.session_state['f'] = f 
            st.session_state['fs'] = fs
            st.session_state['fp'] = fp
            st.session_state['Qm'] = Q
            
        with cols[2]:
            st.latex(r'\gamma_0 = \sqrt{\pi^2 + \frac{8k^2}{1 - k^2}}')
            st.latex(r'C_p = \epsilon^T (1 - k^2) \frac{A}{2l}')
            st.latex(r'C_s = \frac{8k^2}{\pi^2 (1 - k^2)} C_{p+}')
            st.latex(r'L_s = \frac{1 - k^2}{2G_f^2 k^2 v_a^2 C_{p+}}')
            st.latex(r'R_s = \frac{1}{Q_m} \sqrt{\frac{L_+}{C_+}}')
            
            st.latex(r'f_s = \frac{1}{2\pi\sqrt{L_sC_s}}')
            st.latex(r'f_p = \frac{1}{2\pi\sqrt{L_sC_s(1+\frac{C_s}{Cp})}}') 
            B = st.session_state['Cp']*st.session_state['f']
            st.latex(r'B = fC_p = ' + disp_ans(B))
            B0 = B*st.session_state['l'] /(st.session_state['A']*st.session_state['Gf'])
            st.latex(r'B_0 = \frac{Bl}{AG_f}=' + disp_ans(B0))
            
            
            e33 = st.session_state['e33']
            k31_sq = st.session_state['k31_sq']
            gamma = st.session_state['gamma'] 
            kappa_0 = st.session_state['kappa_0']
            va = st.session_state['va']
            Qm = st.session_state['Qm']
            kappa_bar = np.pi*gamma/(np.pi+gamma)
            B02 = e33*(1-k31_sq)*kappa_0*va/(4*np.pi)
            st.latex(r'B_0 = \epsilon_{33}^T \left( 1 - k_{31}^2 \right) \frac{\kappa_o v_a}{4 \pi}')
            st.latex(r'= ' + disp_ans(B02))
            
            R0 = np.pi/(2*Qm*k31_sq*e33*va)
            st.latex(r'R_0 = ' + disp_ans(R0) + r'\text{Ω}')
            
            FOMm = 4*Qm*k31_sq/((1-k31_sq)*np.pi**2*kappa_bar)
            st.latex(r'FOM_M =' + disp_ans(FOMm))
            
            
        sf = np.log10(.99*float(fp.evalf(5)))
        ef = np.log10(1.01*float(fs.evalf(5)))
        
        #start_freq = st.number_input(label=r'$f_{min}$', format='%.3e', value=sf)
        #end_freq = st.number_input(label=r'$f_{max}$', format='%.3e', value=ef)
        
        ckt_net = (R(st.session_state['Rs']) + \
                   L(st.session_state['Ls']) + \
                       C(st.session_state['Cs'])) | C(st.session_state['Cp'])
        vf = np.logspace(sf,ef, 9000)

        fig, axes = plt.subplots(1)
        formatter0 = EngFormatter(places=2, unit='MHz', sep='')
        formatter_y = EngFormatter(places=2, unit='S', sep='')
        
        #ckt_net.Y(j2pif).magnitude.plot(vf/1e6, axes=axes)
        axes.plot( vf/1e6, ckt_net.Y(j2pif).magnitude.evaluate(vf))
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_ylabel('Admittence')
        axes.xaxis.set_major_formatter(formatter0) 
        axes.yaxis.set_major_formatter(formatter_y)
        axes.xaxis.set_major_locator(MaxNLocator(nbins=5))
        #axes.xaxis.xticks(rotation=45)
        axes.tick_params(axis='x', labelrotation=25)
        #axes.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))

        # Set minor ticks and disable them if necessary
        axes.minorticks_off()
        st.pyplot(fig=fig)
        


with calcs:
        s = r'I_{Lmaxo}^S = v_a \frac{d_{31}}{s_{11}^E} S_{\text{max}} \sin(\kappa_o)'
        st.latex(s)
        
        s2 = r'I_{Lmaxo}^T =v_a d_{31} T_{\text{max}} \cot\left(\frac{\kappa_o}{2}\right)'
        st.latex(s2)
        
        s3 = r'I_{Lmaxo}^E = v_a k_{31}^2 \epsilon_{33}^T E_{\text{max}} \tan(\kappa_o)'
        st.latex(s3)
        s4 = r'I_{Lmaxo}^H =\sqrt{\frac{4Qmk_{31}^2\epsilon_{33}^T v_a}{\pi H} '+\
            'E_{\text{max}}}'
        st.latex(s4)
