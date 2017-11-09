#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:44:46 2017

@author: danjampro
"""

import numpy as np
from . import geometry

#==============================================================================

def MC_density(data, eps, thresh, rms, N=10000):
    
    '''Monte-carlo measurement of point density'''
    
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    X, Y = np.meshgrid(np.arange(-int(eps), int(eps)+1), np.arange(-int(eps), int(eps)+1))
    cutout = X**2 + Y**2 < eps**2
    results = []
    x0s = []
    y0s = []
    for i in range(N):
        x0 = int( np.random.uniform(0, data.shape[1]-2*eps-5) )
        y0 = int( np.random.uniform(0, data.shape[0]-2*eps-5) )
        results.append( np.sum((data[y0:y0+cutout.shape[0],
                                    x0:x0+cutout.shape[1]])[cutout]>
                                    (rms[y0:y0+cutout.shape[0],
                                    x0:x0+cutout.shape[1]])[cutout]*thresh))
        x0s.append(x0)
        y0s.append(y0)
        
    return np.array(results), cutout, np.array(x0s), np.array(y0s)



def estimate_minpts(kappa, eps, rms, tmin, tmax=np.inf):
    
    '''
    Calculate number of points required to have confidence kappa of not occuring 
    due to noise
    
    Paramters
    ----------
    kappa: Confidence above noise of core point occurance.
    rms: RMS level of the data.
    tmin: Lower brightness threshold.
    tmax (optional): Upper brightness theshold. Default is np.inf.
    '''
    
    from scipy.special import erf
        
    #Probability of a data point lying within threshold
    Pthresh = 0.5 * ( erf(tmax/(np.sqrt(2)*rms)) - erf(tmin/(np.sqrt(2)*rms)) )
    
    #Total number of pixels in circle of eps
    Ntot = _pixels_in_circle(eps)
    
    #Standard deviation for binomial Gaussian
    sigN = np.sqrt(Pthresh * (1-Pthresh) * Ntot)
    
    #Average number of pixels within eps within threshold for binomial Gaussian
    Nbar = Pthresh * Ntot
    
    #print(Pthresh, Ntot, Nbar, sigN, P0, minpts)
    minpts = int( Nbar + (kappa * sigN) )
    
    return minpts



def _pixels_in_circle(eps):
    '''Return the number of pixels within an epsilon radius'''
    return np.sum(geometry.unit_tophat(eps))

#==============================================================================