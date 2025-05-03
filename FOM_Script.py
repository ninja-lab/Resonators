#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:56:45 2024

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
#import math
from math import sqrt
from sympy import symbols, pi, sqrt, solve, nonlinsolve, Rational, nsolve
from sympy.printing import latex
from scipy.interpolate import interp1d
from FOM_Script_Helpers import *
eps_0 = constants.epsilon_0

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
e_mat = Matrix(e)
s_mat = Matrix(S)
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



for var in varss: 
    if str(var) not in st.session_state:
        st.session_state[str(var)] = None

if 'sym_sols' not in st.session_state:
    st.session_state.sym_sols = {str(var): None for var in varss}





constants, geometry, calcs = st.tabs(["Material Constants", "Geometry", "Calculations"])
with constants:
    with st.expander('Material Constants for __ GaN'):
        st.title('Material Constants for __ GaN')
    
        C_scientific = Matrix(C).applyfunc(formatter)
        st.header(r'Stiffness Matrix $\mathbf{C} \,\frac{\text{N}}{\text{m}^2}$')
        st.latex(latex(C_scientific)) #m^2 / N   
        
        S_scientific = Matrix(S).applyfunc(lambda x: '{:.2e}'.format(float(x)))
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
        va = 1/sqrt(rho*s11)
        va_str = r'v_a = \frac{1}{\sqrt{\rho s_{11}^E}}= '+\
            f'{va:.0f}' + r'\, \frac{\text{m}}{\text{s}}'
        st.latex(va_str)
    
    with col2:
        #ans = 8.9*eps_0
        e33 = eps_mat[2,2]
        eps = r'\epsilon_{33}^T = \epsilon_{r,33}\cdot \epsilon_0=' +\
              disp_ans(e33) + r'=\frac{\text{F}}{\text{m}}'   
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
    st.divider()
    option = st.selectbox("Configuration?", 
                          (r'S0, no bottom electrode',r'S0, bottom electrode',
                           r'A0, no bottom electrode', r'A0, bottom electrode'),
                          key='electrode_config')
    st.write('k^2')
    hGaN = 2.5e-6 #thickness [m]
    lmda = 4e-5 # wavelength [m]
    
    
    if option == r'S0, no bottom electrode':
        ksq_perc = ksq_functions['S0_no_elec'](x)
        ksq_perc = ksq_functions['S0_no_elec'](0)
    elif option == r'S0, bottom electrode':
        ksq_perc = ksq_functions['S0_bot_elec'](0)
    elif option == r'A0, no bottom electrode':
        ksq_perc = ksq_functions['A0_no_elec'](0)
    elif option == r'A0, bottom electrode': 
        ksq_perc= ksq_functions['A0_bot_elec'](0)
    
with geometry: 

    with st.container():
        cols = st.columns(2)
        with cols[0]:
    
            for i, var in zip(range(len(varss)), varss):
    
                st.number_input(label = '$'+str(var)+ r'\,\, [\text{'+ var.unit + r'}]$',
                                value=None,
                                min_value=var.min,
                                max_value=var.max,
                                step=var.step,
                                format=var.format if var.format !='' else None,
                                key=str(var),
                                on_change=cau,
                                args=(str(var),varss))


            
        with cols[1]:

            for eq in eqs:
                
                st.latex(myprint2(eq))
            if 'results' not in st.session_state:
                results = st.empty().container()
                st.session_state['results'] = results
  
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
