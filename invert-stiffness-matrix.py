#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:08:09 2024

@author: erik
"""
import numpy as np
from sympy import Matrix
from sympy.printing import latex
from IPython.display import display
# Define the stiffness matrix C_ij, values from Popa 
C = np.array([
    [390, 145, 106, 0, 0, 0],
    [145, 390, 106, 0, 0, 0],
    [106, 106, 398, 0, 0, 0],
    [0, 0, 0, 105, 0, 0],
    [0, 0, 0, 0, 105, 0],
    [0, 0, 0, 0, 0, 122.5]
])*1e9

# Invert the matrix to get the compliance matrix S_ij
S_ij = np.linalg.inv(C)
#display(Matrix(np.round(S_ij, 6)))
S_scientific = Matrix(S_ij).applyfunc(lambda x: '{:.2e}'.format(float(x)))
display(S_scientific) #m^2 / N 

def g(x):
    #if (-.1 < float(x.evalf()) and float(x.evalf()) < .1):
    if abs(float(x.evalf())) < 1e-10:
        return '0'
    else:
        return f'{float(x.evalf(8)):.2e}'
C_scientific = Matrix(C).applyfunc(g)

latex(C_scientific) #m^2 / N   
print(C_scientific)