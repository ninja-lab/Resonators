#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 08:42:46 2024

@author: erik
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:08:09 2024

@author: erik
"""
import numpy as np
from sympy import Matrix, Float
from sympy.printing import latex
from IPython.display import display
# Define the stiffness matrix C_ij, values from Popa 
C = np.array([
    [390, 145, ],
    [145, 0, ],
    ])*1e9
def formatter(x):
    return Float(x, 3)
# Invert the matrix to get the compliance matrix S_ij
S_ij = np.linalg.inv(C)
#display(Matrix(np.round(S_ij, 6)))
S_scientific = Matrix(S_ij).applyfunc(lambda x: '{:.2e}'.format(float(x)))
print('this prints as desired:')
display(S_scientific) #m^2 / N 

def g(x):
    #shouldn't be necessary since 0 handles fine above
    if abs(float(x.evalf())) < 1e-10:
        return '0'
    else:
        return f'{float(x.evalf(8)):.2e}'
C_scientific = Matrix(C).applyfunc(formatter)
print()
print('this is not print in scientific notation:')
display(C_scientific) 
print(C_scientific)