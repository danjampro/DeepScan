#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:11:40 2017

@author: danjampro
"""
import numpy as np
from scipy.special import gamma, gammainc, gammaincinv
from scipy.optimize import minimize

#==============================================================================

def profile(r, Ie, re, n):
    '''
    Evaluate the intensity at r.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    b = sersic_b(n)
    return Ie * np.e**( -b * ((r/re)**(1./n)-1) )


def profile_SB(r, ue, re, n):
    '''
    Evaluate the surface brightness at r.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    return ue + 2.5*sersic_b(n)/np.log(10) * ((r/re)**(1./n)-1)


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
    if hasattr(n, '__len__'):
        n = np.array(n, dtype='float')
        _ = np.isfinite(n)
        out = np.zeros_like(n)
        out[_] = gammaincinv( 2*n[_], 0.5 ) 
        out[~_] = np.nan
        return out
    else:
        if np.isnan(n): return np.nan
        return gammaincinv( 2*n, 0.5 ) 
    
    
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
        Effective (half-light) radius.
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
        Effective (half-light) radius.
    rk : float
        Kron radius integrated to 2.5*re.
        
    Returns
    -------
    float:
        Sersic index.
    '''
    
    return minimize(_index_re_kron, x0=1., args=(re, rk), method='Nelder-Mead',
                    options={'xatol':xatol}).x[0]
    
#==============================================================================

def mag2meanSB(mag, re, q):
    '''
    Converts total magnitude to mean SB within Re.
       
    Parameters
    ----------
    mag : 
        Total magnitude [mag].   
    re : 
        Effective (half light) radius [arcsec].       
    q : 
        Axis ratio (b/a).
    
    Returns
    -------
    float 
        Average surface brightness within th effective radius [mag arcsec-2].
    '''
    return mag + 2.5*np.log10(np.pi*q*re**2) - 2.5*np.log10(0.5)


def meanSB2mag(uae, re, q):
    '''
    Converts mean SB within Re to total magnitude.
       
    Parameters
    ----------
    uae : 
        Average surface brightness within the effective radius [mag arcsec-2].    
    re : 
        Effective (half light) radius [arcsec].        
    q : 
        Axis ratio (b/a).
    
    Returns
    -------
    float 
        Total magnitude.
    '''
    return uae - 2.5*np.log10(np.pi*q*re**2) + 2.5*np.log10(0.5)


def effectiveSB2mag(ue, re, n, q):
    '''
    Converts SB at Re to total magnitude.
       
    Parameters
    ----------
    ue : 
        Surface brightness at the effective radius [mag arcsec-2].    
    re :
        Effective (half light) radius [arcsec].   
    n : 
        Profile index.   
    q : 
        Axis ratio (b/a)
    
    Returns
    -------
    float 
        Total magnitude.
    
    '''
    b = sersic_b(n)
    _ = 2*np.pi*re**2*np.e**b*n*b**(-2*n)*gamma(2*n)*q
    return -2.5*np.log10(_) + ue


def mag2effectiveSB(mag, re, n, q):
    '''
    Converts total magnitude to SB at Re.
       
    Parameters
    ----------
    mag :
        Total magnitude [mag].
    re : 
        Effective (half light) radius [arcsec].
    n : 
        Profile index.   
    q :
        Axis ratio (b/a)
    
    Returns
    -------
    float
        Surface brightness at the effective radius [mag arcsec-2].    
    '''
    b = sersic_b(n)
    _ = 2*np.pi*re**2*np.e**b*n*b**(-2*n)*gamma(2*n)*q
    return mag + 2.5*np.log10(_) 


def effectiveSB2SB0(ue, n):
    '''
    Convert SB at Re to central SB.
    
    Parameters
    ----------
    ue : 
        Surface brightness at the effective radius [mag area-1].     
    n : 
        Profile index.   

    Returns
    -------
    float
        Central surface brightness [mag area-1].
    
    '''
    b = sersic_b(n)
    return ue - 2.5*b/np.log(10)


def meanSB2effectiveSB(uae, re, n, q):
    '''
    Convert mean SB within Re to SB at Re.
    
    Parameters
    ----------
    uae : 
        Average surface brightness within the effective radius [mag area-1].    
    re : 
        Effective (half light) radius.  
    n : 
        Profile index.  
    q : 
        Axis ratio (b/a).
    
    Returns
    -------
    float:
        Surface brightness at the effective radius [mag area-1].    
    '''
    mag = meanSB2mag(uae, re, q)
    return mag2effectiveSB(mag, re, n, q)

#==============================================================================

def h2Re(h, n):
    '''
    Convert exponential scale length to effective radius.
    
    Parameters
    ----------
    h :
        Exponential scale size.
    n : 
        Profile index.  
    
    Returns
    -------
    float
        Effective (half-light) radius.
    '''
    return h*sersic_b(n)**n

def Re2h(re, n):
    '''
    Convert effective radius to exponential scale length .
    
    Parameters
    ----------
    h :
        Effective (half-light) radius.
    n : 
        Profile index.  
    
    Returns
    -------
    float
        Exponential scale length.       
    '''
    return re/sersic_b(n)**n

#==============================================================================

