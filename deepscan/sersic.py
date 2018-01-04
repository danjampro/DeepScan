#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:11:40 2017

@author: danjampro
"""

import numpy as np
from scipy.special import gamma, gammainc, gammaincinv
from scipy.optimize import minimize


def profile(r, Ie, re, n):
    '''
    Evaluate the Sersic profile at r
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    b = sersic_b(n)
    return Ie * np.e**( -b * ((r/re)**(1./n)-1) )



def sersic_b(n):   
    '''
    Calculate beta for a given Sersic index.
    
    Parameters
    ----------
    n : float
        Sersic index.
    
    Returns
    -------
    float:
        Sersic b value.
    '''
    
    b = gammaincinv( 2*n, 0.5 ) 
    
    return b
    
    

def isophotal_radius(uiso, ue, re, n, b=None):
    
    '''Calculate the isophotal radius.
    
    Parameters
    ----------
    uiso : float
        Isophotal SB.
    ue : float
        Mean SB within effective radius.
    re : float
        Effective radius.
    n : float
        Sersic index.
    b : float (optional)
        Sersic b.
        
    Returns
    -------
    float:
        Isophotal radius.
    '''
    
    b = sersic_b(n) if (b is None) else b
    
    A = (uiso-ue) * np.log(10)/(2.5*b) + 1
    
    return re * A**n


def _index_re_kron(n, re, rk):
    
    '''Minimise this to obtain the Sersic index.
    
    Parameters
    ----------
    n : float
        Sersic index.
    re : float
        Effective radius.
    rk : float
        Kron radius integrated to 2.5*re.
        
    Returns
    -------
    float:
        Number to be minimized.
    '''
            
    #This is the integration radius of SExtractor. 
    #Values of Graham & Driver (2005) can be obtained by setting a larger R
    R = 2.5 * rk 

    b = sersic_b(n)
    
    x = b*(R/re)**(1./n)
    
    R0 = rk/re
    R1 = (1./b)**n * gamma(3*n)*gammainc(3*n,x)/(gammainc(2*n,x)*gamma(2*n))
    
    return abs(R0-R1)


def index_re_kron(re, rk, xatol=0.1):
    '''
    Calculate the Sersic index from effective & Kron radii.
    
    Parameters
    ----------
    re : float
        Effective radius.
    rk : float
        Kron radius integrated to 2.5*re.
        
    Returns
    -------
    float:
        Sersic index.
    '''
    
    return minimize(_index_re_kron, x0=1., args=(re, rk), method='Nelder-Mead',
                    options={'xatol':xatol}).x[0]
    
    

def effective_SB(mtot, re, n, b=None):
    '''
    Calculate the SB at the effective radius.
    
    Parameters
    ----------
    mtot : float
        Total magnitude.
    re : float
        Effective radius.
    n : float
        Sersic index.
    b : float (optional)
        Sersic b.
    
    Returns
    -------
    float:
        SB at the effective radius.
    '''
    b = sersic_b(n) if (b is None) else b
    
    A = 2 * np.pi * re**2 *n * np.e**b / b**(2*n)
    
    return mtot + 2.5*np.log10( A * gamma(2*n) )


def average_effective_SB(ue, n):
    '''
    Average surface brightness within the effective radius.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    b = sersic_b(n)
    f = n * np.e**b * gamma(2*n) / b**(2*n)
    return ue - 2.5*np.log10(f)
    
    
def magnitude(ue, re, n):
    '''
    Total magnitude.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    ueav = average_effective_SB(ue, n)
    return ueav - 2.5*np.log10( 2*np.pi*re**2 ) 


def ue2SB0(ue, n):
    '''
    Convert central SB to effective SB.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    b = sersic_b(n)
    return ue - 2.5*b/np.log(10)



