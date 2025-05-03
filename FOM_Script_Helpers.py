#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:08:20 2024

@author: erik
"""
from sympy.printing import latex
from sympy.core.symbol import Symbol
import sympy 
from matplotlib.ticker import  EngFormatter
import math 
from statistics import median
import numpy as np 
from sympy import symbols, pi, sqrt, solve, nonlinsolve, Rational, nsolve
from sympy import Float 
import streamlit as st
import math 

class variable(sympy.core.symbol.Symbol):
    #def __new__(self, name, *args, min_value = None, max_value=None, step=None, **assumptions):
        #obj = Symbol.__new__(self, name, **assumptions )
    def __new__(cls, name, *args, min_value=None, max_value=None, step=None, **assumptions):
        obj = Symbol.__new__(cls, name, **assumptions)
        obj.format = args[1] if len(args)>1 else None
        obj.unit = args[0] if len(args)>0 else ''
        obj.min = min_value
        obj.max = max_value
        obj.step = step
        obj.guess = median(np.logspace(np.log10(obj.min), np.log10(obj.max)))
        obj.value = None 
        return obj
    
    '''
    def __str__(self):
        if self.unit is not None: 
            return super().__str__() + f' {self.unit}'
        else: 
            return super().__str__()
    '''
    
    def set_value(self, value):
        self.value = value
    
    def get_value(self):
        return self.guess if self.value is None else self.value
        
    def show_self(self):
        #try:
        return latex(self) + f' = {self.value:{self.format}} {self.unit}'
        #except ValueError:
            
     
        
fres_min = 1e3
fres_max = 10e9
x_min = .01
x_max = .99
lmbda_min = 2.0
lmbda_max = 10.0 
hGaN_min = 1.0e-6
hGaN_max = 5.0e-6
a_min = lmbda_min*3/4
a_max = lmbda_max*3/4

x = variable('x', r'\,', '%.3f', min_value=x_min, max_value=x_max,step=.01, real=True, positive=True)
hGaN = variable(r'h_{GaN}','m', '%.2e', min_value=hGaN_min, max_value=hGaN_max, step=.1, real=True, positive=True)
lmbda = variable(r'\lambda','m', '%.2e', min_value = lmbda_min, max_value=lmbda_max, step=.01, real=True, positive=True)
a = variable('a','m','%.2e', min_value = a_min, max_value=a_max , real=True, positive=True)
fres = variable('f_{res}', 'Hz', '%.2e',min_value=fres_min, max_value=fres_max, step=fres_min, real=True, positive=True )


varss = [x, hGaN, lmbda, a, fres]

eq1 = fres*lmbda - 7254
eq2 = hGaN/lmbda - x
eq3 = 2*a - 3*lmbda/2
eqs = [eq1, eq2, eq3]


def cau(*args):
    
    '''
    Parameters
    ----------
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    since nonlinsolve has no tolerance on accepting old solutions, 
    try a numeric solve when the nonlinsolve comes up empty. 
    Initial guesses are what is known already. 
    '''
    
    var = args[0]
    varss = args[1]
    def zipsys(varss, inputs):
        '''
        Parameters
        ----------
        varss : list of symbols in the system
        inputs : {var, value} where value is a float or None

        Returns
        -------
        To increase the chance of solver being able to use prior solutions, 
        use symbolic result from sym_sols if one is there. 

        But, don't replace the index sym for which the callback was called
        since that has changed from user. 
        
        There may be numbers in number_inputs already, and the symbolic form
        is already in sym_sols. 
        '''
        given = []
        print(f'symsols: {st.session_state.sym_sols}')
        print(var)
        for s, v in inputs.items():
            print(f'str s: {s}')
            print(f'v: {v}')
            if str(s) != var: #not appending the var for which callback was called 
                try:
                    print(f'appending var {s} = {st.session_state.sym_sols[str(s)]} ')
                    #already rational: 
                    given.append(s - st.session_state.sym_sols[str(s)])  
                except TypeError:
                    pass

            else: #don't think this should ever fail with TypeError because
            #this catches the new one 
                try:
                    given.append(s - Rational(v))
                except TypeError:
                    print('caught type error!')
                    #print(s)
                    #print(v)
                    pass                 
        return given
    
    def my_remove(var, inputs):
        '''
        
        Parameters
        ----------
        var: string from  compute_and_update (cau) callback arg
        inputs: {var: value}, where value is from number_inputs 
        Returns
        -------
        inputs with a value removed that isn't var. 
        
        It would be nice to remove the var that would leave the most solutions
        . 
        You could try more than one, and stick with the one that leaves 
        the most answers. Or, use the eqs, pick one from an equation 

        '''
   
        for s in inputs.keys():
            if inputs[s] != None and str(s) != var:
                print(f'removing {s} = {inputs[s]}')
                inputs[s] = None
                st.session_state.sym_sols[str(s)] = None 
                break
        return inputs
    

    inputs = {var: st.session_state[str(var)] for var in varss} #from number_inputs
    print(f'inputs: {inputs}')
    #the total system is the constraints plus what is input 
    #would it be better to substitute in what is known instead of adding 
    #equations? 
    mysys = eqs + zipsys(varss, inputs)  
    print(mysys)
    ans = filter_solutions(nonlinsolve(mysys, varss ), varss) 
    print(f'filtered before loop: {ans} \n')
    if len(ans)==0:
        for j in range(len(varss)): #only try len(varss) times 
            
            print('caught empty set')
            #remove an input that is not the newest and also isn't already None
            inputs = my_remove(var, inputs)
            mysys = eqs+ zipsys(varss, inputs)  
            print(mysys)
            ans = filter_solutions(nonlinsolve(mysys, varss ),varss) 
            if len(ans) > 0:
                break 

    print(f'filtered: {ans} \n')
    print(f'num ans: {len(ans)}')
    for sol in ans: #what if there is no sol?  
        
        for var, res, i in zip(varss, sol, range(len(varss))):
            #if there are no symbols in the expression, save it 
            with st.session_state['results']:
            
                if len(res.free_symbols) == 0:
                    #we're in a callback that won't be called until a number_input 
                    #changes. results is an already created container in an st.empty
                    #object. 
                    st.session_state.sym_sols[str(var)] = res #keep it rational 
                    st.session_state[str(var)] = float(res)
                    var.set_value = float(res)
                    st.latex(latex(var) +' = '  + process_res(var, res ))
                    
                else: 
                    st.latex(latex(var) +' = '  + latex(res.evalf(4)))
                    st.session_state[str(var)] = None 
        break #cludgy way to not display multiple results for now 
                
    return 


def disp_ans(ans):
    #might do the same thing as latex(sym.evalf(3))
   try:
       ans1, ans2 = power_of_10(ans)
       return f'{ans1:.2f}' +  r'\cdot'+ r'10^{'+ f'{ans2}' + r'}\,' 
   except ValueError as e:
       return ans

def formatter(x):
    return Float(x, 3)


def myprint(*args):
    '''  
    the first argument is the only "lhs"
    subsequent arguments are rhs-s
    '''
    
    def helper(args):
        if len(args) == 0:
            return ''
        else: 
            return f'= {latex(args[0])}' + helper(args[1:])
        
    s = f'{latex(args[0])} {helper(args[1:])}' 
    return s 
def myprint2(eq):
    #if isinstance(eq2.func, sympy.core.add.Add):
    if eq.func == sympy.core.add.Add:
        lhs = eq.args[0]
        rhs = eq - lhs
        if lhs.is_number and lhs < 0:
            return myprint(-1*lhs, rhs)
        try: 
            lhs_exp = lhs.args
            if any([isinstance(x, sympy.core.numbers.NegativeOne) 
                    for x in lhs_exp]):
                return myprint(-1*lhs, rhs)
        except IndexError:
            pass            
        return myprint(lhs, -1*rhs) 


def filter_solutions(solutions, symbols):
    def satisfies_assumptions(solution, symbols):
        for sol, symbol in zip(solution, symbols):
            # Check assumptions only for numeric elements
            if sol.is_number:
                assumptions = symbol.assumptions0
                if assumptions.get('positive', False) and not sol.is_positive:
                    return False
                if assumptions.get('negative', False) and not sol.is_negative:
                    return False
                if assumptions.get('real', False) and not sol.is_real:
                    return False
                if sol > symbol.max:
                    return False
                if sol < symbol.min:
                    return False 
        return True

    return {sol for sol in solutions if satisfies_assumptions(sol, symbols)}


def count_sig_figs(digits):
    '''Return the number of significant figures of the input digit string'''

    integral, _, fractional = digits.partition(".")

    if fractional:
        return len((integral + fractional).lstrip('0'))
    else:
        return len(integral.strip('0'))

def power_of_10(num):
    # Get the power of 10
    power = int(math.log10(abs(num)))
    # Adjust the number accordingly
    adjusted_num = num / (10 ** power)
    return adjusted_num, power

def process_res(var, res):
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
    
    #n, e = power_of_10(res.evalf())

   
    #return  formatter.format_eng(round(float(res.evalf()),3)) + var.unit
    expr = formatter.format_eng(res.evalf())
    letter = expr[-1]
    num = expr[:-1]
    
    return  num + r'\,\, \text{'+ letter + var.unit + r'}'


