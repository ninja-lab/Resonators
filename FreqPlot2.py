#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:16:50 2024

@author: erik
"""

import streamlit as st 
import numpy as np 
from scipy import constants
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, EngFormatter
import math

def power_of_10(num):
    # Get the power of 10
    power = int(math.log10(abs(num)))
    # Adjust the number accordingly
    adjusted_num = num / (10 ** power)
    return adjusted_num*10, power-1
 
eps_0 = constants.epsilon_0
pi = np.pi
sqrt = np.sqrt
log = np.log 
exp = np.exp

data = {
    (0, 0): (5.7832, 10.215, 0.1148, 0.4868),
    (0, 1): (14.682, 21.26, 0.0613, 0.5273),
    (0, 2): (26.375, 34.877, 0.0444, 0.5334),
    (1, 0): (30.471, 39.771, 0.0428, 0.5315),
    (0, 3): (40.707, 51.030, 0.0360, 0.5307),
    (1, 1): (49.219, 60.829, 0.0341, 0.5282),
    (0, 4): (57.583, 69.666, 0.0303, 0.5284),
    (1, 2): (70.850, 84.583, 0.0283, 0.5265),
    (2, 0): (74.887, 89.104, 0.0278, 0.5260),
    (0, 5): (76.939, 90.739, 0.0258, 0.5278),
    (1, 3): (95.278, 111.02, 0.0239, 0.5271),
    (0, 6): (98.726, 114.21, 0.0223, 0.5290),
    (2, 1): (103.50, 120.08, 0.0232, 0.5268),
    (1, 4): (122.43, 140.11, 0.0204, 0.5296),
    (0, 7): (122.91, 140.06, 0.0193, 0.5316)
}

if 'd_sld' not in st.session_state:
    st.session_state['d_sld'] = 10

if 't_sld' not in st.session_state:
    st.session_state['t_sld'] = 10

R = st.session_state.d_sld/2 #needs to get informed from diameter slider
delta_R=.05*R    #this is made up for now 
'''
total strain 

'''
MoS2 = {'Ey': 240*1e9 , 'er':delta_R/R, 'rho':5060, 'v':.125 }
graphene={'Ey': 1e12 , 'er':delta_R/R, 'rho':2270, 'v':.19 }
matl_dict = {'MoS2': MoS2, 'Graphene':graphene}
st.selectbox('Material', ('MoS2', 'Graphene'), key='matl_str')#, label='Material')

matl = matl_dict[st.session_state['matl_str']]

@st.cache_data
def calc_f(Vg, t, g, R, Ey, er, rho, **kwargs):
    '''
    

    Parameters
    ----------
    Vg : voltage bias [V]
    t : thickness [nm]
    g : gap [nm]
    R : radius [um] 
    Ey : young's modulus [Pa]
    er : radial strain = ΔR / R
    rho : mass density [kg/m^3]
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    resonant frequency 

    '''
    g_m = g*1e-9
    t_m = t*1e-9
    R_m = R*1e-6
    return (1/(2*pi))*sqrt((2.4**2*Ey*er/(rho*R_m**2))-(eps_0/(.813*rho*t_m*g_m**3))*Vg**2)
@st.cache_data
def calc_D(t, **kwargs):
    #t is in meters 
    return kwargs['Ey']*t**3/(12*(1-kwargs['v']**2))
col1, col2 = st.columns(2)
with col1: 
    

    st.latex(r'''
    f_{\text{res,FC}} = \frac{1}{2\pi} \sqrt{\frac{2.4^2 E_Y \varepsilon_r}{\rho_{m} R^2} - \frac{\varepsilon_0}{0.813 \rho_{m} t g^3} V_G^2}
    
        ''')
    
    st.slider('$V_G \,\, [V]$', min_value=0.0, max_value=2.0, step=.1,
                   key='Vg_sld')  # 👈 this is a widget
    st.slider('$g \,\, [nm]$', min_value=10, max_value=150, step=5,
                   key='g_sld')  # 👈 this is a widget
    
    t = np.logspace(0,2)*.7 #thickness in nm
    g = 10e-9
    #for the plot of freq vs thickness, with variable Vg, R, g
    f = calc_f(st.session_state.Vg_sld, t, 
               st.session_state.g_sld, R, **matl)  
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t[0]*.2, t[-1]*10)
    ax.plot(t, f/1e6, label=r'$ f_0 $')
    ax.plot(t, f*1.59/1e6, label = r'$ 1.59 \cdot f_0 $')
    ax.plot(t, f*2.14/1e6, label = r'$ 2.14 \cdot f_0 $')
    major_formatter= EngFormatter(unit='nm', places=0, sep='')
    ax.xaxis.set_major_formatter(major_formatter)  
    major_formatter= EngFormatter(unit='', places=0, sep='')
    ax.yaxis.set_major_formatter(major_formatter)  
    ax.set_xlabel('2D thickness')
    ax.set_ylabel('Resonant Frequency [MHz]')
    ax.set_title('Effect of Thickness on Resonant Frequency')
    ax.set_ylim(.1, 1000)
    ax.legend()
    st.pyplot(fig=fig)
    
    Vg  = np.linspace(0,2) #bias voltage
    
    #for the plot of freq vs thickness, with variable Vg, R, g
    f = calc_f(Vg, st.session_state.t_sld, 
               st.session_state.g_sld, R, **matl)  
    fig, ax = plt.subplots()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xlim(t[0]*.2, t[-1]*10)
    ax.plot(Vg, f/1e6, label=r'$f_0 $')
    #ax.plot(t, f*1.59/1e6, label = r'$1.59 \cdot f_0 $')
    #ax.plot(t, f*2.14/1e6, label = r'$2.14 \cdot f_0 $')
    #major_formatter= EngFormatter(unit='nm', places=0, sep='')
    #ax.xaxis.set_major_formatter(major_formatter)  
    major_formatter= EngFormatter(unit='', places=2, sep='')
    ax.yaxis.set_major_formatter(major_formatter)  
    ax.set_xlabel('Voltage Bias')
    ax.set_ylabel('Resonant Frequency [MHz]')
    ax.set_title('Effect of Bias on Resonant Frequency')
    #ax.set_ylim(.1, 1000)
    ax.legend()
    st.pyplot(fig=fig)

with col2: 

  
    ans = calc_D(st.session_state.t_sld*1e-9, **matl)
    if 'D' not in st.session_state:
        st.session_state['D']=ans
    ans1, ans2 = power_of_10(ans)
    st.latex(r'\text{flexural rigidity D}')
    lat = r'D = \frac{E_Y t^3}{12(1 - \nu^2)} = '
    my_str = lat + f'{ans1:.2f}' r'\cdot'+ \
        r'10^{'+ f'{ans2}' + r'}\,\text{N}\cdot\text{m}'
    st.latex(my_str)  
    @st.cache_data
    def f2(x, a, b, g, d):
        return a + (b-a)*exp(-g*exp(d*log(x)))
    @st.cache_data
    def calc_f2(t=10, gamma=.5, d=2, mode=(0,0), **kwargs ):
        '''
        

        Parameters
        ----------
        t : thickness in nm
        d : diameter in um
        gamma : tension in N/m
        Returns
        -------
        Ta**2/D or gamma*d**2/(4D) indicates 
        how far in to tension dominant limit (membrane)
        or modulus dominant (plate) the drum is. 
        Plug that number into f(x) to get kd/2 or k*a
        Plug kd/2 into expression for f 
        '''
        d_m = d*1e-6 #convert um to m
        t_m = t*1e-9 #convert nm to m
        D = calc_D(t_m, **kwargs)
        x = gamma*(d_m/2)**2/D
        kd = f2(x, *data[mode])
        rho_a = kwargs['rho']*t_m
        return (kd/(4*pi))*sqrt((16*D)*(kd**2/4+gamma*d_m**2/(4*D))/(rho_a*d_m**4))  
    def calc_memb_f(t=10, gamma=.5, d=2,mode=(0,0), **kwargs):

        d_m = d*1e-6 #diameter in meters
        t_m = t*1e-9 #thickness in meters
        D = calc_D(t_m, **kwargs)
        x = gamma*(d_m/2)**2/D
        kd = f2(x, *data[mode])
        rho_a = kwargs['rho']*t_m
        return kd*sqrt(gamma/rho_a)/(d_m*2*pi)
    def calc_plate_f(t=10, gamma=.5, d=2, mode=(0,0), **kwargs):
        t_m = t*1e-9 #thickness in meters
        d_m = d*1e-6 #diameter in meters
        D = calc_D(t_m, **kwargs)
      
        x = gamma*(d_m/2)**2/D
        kd = f2(x, *data[mode])
        rho_a = kwargs['rho']*t_m
        return kd**2*sqrt(D/rho_a)/(d_m**2*2*pi)   
    

        
    st.slider(r'$\text{thickness} \, t \,\, \text{[nm]}$',
                      min_value=1, max_value=70, step=1,
                      key='t_sld')
    st.slider(r'$\text{diameter} \, d=2R \,\, [\mu\text{m}]$',
                      min_value=2, max_value=50, step=1,
                      key='d_sld')
    st.slider(r'$\text{tension} \, \gamma \,\, [\text{N/m}]$',
                      min_value=.1, max_value=5.0, step=.1,
                      key='gamma_sld')
    st.selectbox('Mode', data.keys(), key='mode_sct' ) 
    st.latex(r'''
    f_0 = \left( \frac{kd}{4\pi} \right) \sqrt{\frac{16D}{\rho_a d^4} \left[ \left( \frac{kd}{2} \right)^2 + \frac{\gamma d^2}{4D} \right]}
        ''')
     
    ans = calc_f2(t=st.session_state.t_sld, gamma=st.session_state.gamma_sld,
                  d=st.session_state.d_sld, mode=st.session_state.mode_sct,
                  **matl)
    ans1, ans2 = power_of_10(ans/1e6)
    my_str = f'= {ans1:.2f}' r'\cdot'+ \
        r'10^{'+ f'{ans2}' + r'}\,\text{[MHz]}'
    st.latex(my_str)  
     
    st.latex(r'''
f(x) = \alpha + (\beta - \alpha) \exp\left(-\gamma \exp\left(\delta \ln(x)\right)\right)
        ''')   

    t2 = np.logspace(0,2)*.7 # thickness in nm
    f3 = calc_f2(t=t2,gamma = st.session_state.gamma_sld,
                 d = st.session_state.d_sld, mode=st.session_state.mode_sct, **matl)
    f4 = calc_memb_f(t=t2,gamma = st.session_state.gamma_sld,
                     d = st.session_state.d_sld, mode=st.session_state.mode_sct, **matl)
    f5 = calc_plate_f(t=t2,gamma = st.session_state.gamma_sld,
                      d = st.session_state.d_sld,mode=st.session_state.mode_sct, **matl)
    f6 = calc_f(st.session_state.Vg_sld, t2, st.session_state.g_sld,
                R, **matl) 
    fig2, ax2 = plt.subplots()
    ax2.plot(t2, f3/1e6)
    ax2.plot(t2, f4/1e6, linestyle='--', label='memb')
    ax2.plot(t2, f5/1e6, linestyle='--', label='plt')
    ax2.plot(t2, f6/1e6, label='bias eq')
    major_formatter= EngFormatter(unit='m', places=0, sep='')
    ax2.xaxis.set_major_formatter(major_formatter)  
    major_formatter= EngFormatter(unit='', places=0, sep='')
    ax2.yaxis.set_major_formatter(major_formatter)  
    ax2.set_xlabel('2D thickness [nm]')
    ax2.set_ylabel('Resonant Frequency [MHz]')
    ax2.set_title('Effect of Thickness on Resonant Frequency')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(.5, 75)
    ax2.legend()
    ax2.set_ylim(5, 5e3)
    st.pyplot(fig=fig2)
    st.write(calc_f(0, .5e-9, g, R, **matl) )
   

        
        
        
        
        
        
        
        
       