# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:17:50 2020

@author: Rania
"""
import numpy as np
from scipy import optimize


#ai.T = ième ligne de C, bi = ième coef de d 
A = np.array([[3.0825,0,0,0],
              [0,0.0405,0,0],
              [0,0,0.0271,-0.0031],
              [0,0,-0.0031,0.0054]])

b = np.array([2671,135,103,19])

C = np.array([[-0.0401,-0.0162,-0.0039,0.0002],
              [-0.1326,-0.0004,-0.0034,0.0006],
              [1.5413,0,0,0],
              [0,0.0203,0,0],
              [0,0,0.0136,-0.0015],
              [0,0,-0.0016,0.0027],
              [0.0160,0.0004,0.0005,0.0002]])

d = np.array([-92.6,-29,2671,135,103,19,10])

W0= [i for i in range(0,7)]

def f(x):
    return 0.5 * np.vdot(np.dot(x,A),x) - np.vdot(b,x)

def grad_f(x):
    return np.dot(A,x) - b 

p0 = np.array([1 for i in range(4)])

def minimisation_p(W, p0, x):
    
    def fct_direction (p):
        return 0.5 * np.vdot(np.dot(p,A),p) + np.vdot(np.dot(A,x)-b, p)
    
    def contraintes(p):
        return np.array([np.vdot(C[i],p) for i in W])
    
    def grad_contraintes(p):
        return np.array([C[i] for i in W])
    
    eq_cons_f = {'type': 'eq','fun' : contraintes,'jac' : grad_contraintes}
    
    return optimize.minimize(fct_direction, p0, method='SLSQP', jac = grad_f, constraints=[eq_cons_f], options={'ftol': 1e-9, 'disp': True}).x
    
    

def calcul_alpha(x,p,W):
    I = [i for i in W0  if (i not in W and np.dot(C[i],p)>0)] 
    alpha = 1 
    j = 0
    for i in I:
        test = d[i]-np.dot(C[i],x)/np.dot(C[i],p)
        if test < alpha: 
            alpha = test 
            j = i 
    return alpha, j 


def QP(x0, p0, f, grad_f, W = W0):
    x = x0
    p= minimisation_p(W, p0, x)
    
    if not p.all() == 0: 
        alpha, j = calcul_alpha(x,p,W)
        x = x + np.dot(alpha, p)
        if alpha < 1: 
            W.append(j)
            
    else :
        
        grad_c_w = np.array([C[i] for i in W])
        Lambda =  np.linalg.solve(-grad_c_w, grad_f(x))
        if Lambda > 0 :
            return x 
        W.pop(Lambda.index(min(Lambda)))
    return QP(x, p0, f, grad_f, W)
                