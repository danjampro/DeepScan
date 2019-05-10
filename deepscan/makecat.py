#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 19:28:39 2019

@author: danjampro

Make catalogue from data, segmap and deepscan.source.Source list.

A Source is just an object that has a "segID" and "slc" attribute.
"""
import time
import numpy as np
import pandas as pd

#==============================================================================

def flux(data, segmap, source):
    '''
    Total flux in segmap.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    return {'flux':source.get_data(data=data,segmap=segmap).sum()}
    

def area(segmap, source):
    '''
    Number of pixels in segmap.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    return {'N100':(segmap[source.slc]==source.segID).sum()}


def half_light(data, segmap, source, quantile=0.5):
    '''
    Estimate quantaties associated with the half-light radius.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    #Segment data in descending order
    sdata = source.get_data(data=data,segmap=segmap)
    sdata[::-1].sort()
    
    #At quantile
    flux = sdata.sum()
    N50 = np.argmin(abs(np.cumsum(sdata)-(flux*quantile)))
    R50 = np.sqrt(N50 /np.pi)
    I50 = sdata[N50]
    I50av = sdata[:N50+1].mean()
    
    #Full source
    N100 = sdata.size
    R100 = np.sqrt(sdata.size / np.pi)
    I100 = sdata[-1]
    I100av = sdata.mean()
            
    return {'N100':N100, 'N50':N50, 'R100':R100, 'R50':R50, 'I50':I50,
            'I50av':I50av, 'I100':I100, 'I100av':I100av, 'flux':flux}
    
#==============================================================================
    
def fit_ellipse(data, segmap, source, power=2, quantile=None):
    
    '''
    Fit ellipse to data.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    #Coordinates
    xs, ys = source.get_coordinates(segmap=segmap)
    
    #Weights
    weights = source.get_data(data=data, segmap=segmap)**power
    if quantile is not None:
        weights[weights<np.quantile(data, quantile)] = 0
        
    #First order moments
    x0 = np.average(xs, weights=weights)
    y0 = np.average(ys, weights=weights)
        
    #Second order moments
    x2 = np.sum( (xs-x0)**2 * weights ) / np.sum(weights)
    y2 = np.sum( (ys-y0)**2 * weights ) / np.sum(weights)
    xy = np.sum( (ys-y0)*(xs-x0) * weights ) / np.sum(weights)
    
    #Handle infinitely thin detections
    if x2*y2 - xy**2 < 1./144:
        x2 += 1./12
        y2 += 1./12
    
    #Calculate position angle
    theta = np.sign(xy) * 0.5*abs( np.arctan2(2*xy, x2-y2) ) + np.pi/2
    
    #Calculate the semimajor & minor axes  
    c1 = 0.5*(x2+y2)
    c2 = np.sqrt( ((x2-y2)/2)**2 + xy**2 )
    arms = np.sqrt( c1 + c2 )
    brms = np.sqrt( c1 - c2 )
    
    return {'xcen':x0, 'ycen':y0, 'q':brms/arms, 'theta':theta,
            'a_rms':arms, 'b_rms':brms}
    
#==============================================================================
#Macro function
    
def MakeCat(data, segmap, sources, verbose=True, sky=0):
    '''
    Produce an output catalogue using measurements based on segments.
    
    Parameters
    ----------
    data : 2D float array
        The data.
        
    segmap : 2D int array
        The segmentation image.
        
    sources : list of source.Source objects
        The sources corresponding to the segmentation image.
        
    verbose : bool
        Print information?
        
    sky : float or 2D array of floats
        The sky level.
    
    Returns
    -------
    pandas.DataFrame
        The measurements catalogue.
        
    Notes
    -----
    These measurements are approximate. It may be necessary to follow-up
    with parametric fitting routines.
    
    flux : The total flux within the segment.
        [ADU]
    
    N100 : The number of pixels within the segment.
        [pixels]
    
    N50 : The number of pixels containing half the flux of the segment.
        [pixels]
    
    R100 : The total (elliptical) radius of the segment.
        [pixels]
        
    R50 : The half-light (elliptical) radius of the segment.
        [pixels]
        
    I50 : Brightness at the half-light radius.
        [ADU]
        
    I50av : Average brightness within half-light radius.
        [ADU]
        
    I100 : Minimum brightness of segment.
        [ADU]
        
    I100av : Average brightness within segment.
        [ADU]
    
    xcen : The flux-weighted x-coordinate of the centre of the segment.
        [pixels]
        
    ycen : The flux-weighted y-coordinate of the centre of the segment.
        [pixels]
        
    q : The flux-weighted axis ratio of the segment.
        
    theta : The position angle of the segment relative to y-axis.
        [radians]
        
    a_rms : The flux weighted RMS of the segment along the semi-major axis.
        [pixels]
        
    b_rms : The flux weighted RMS of the segment along the semi-minor axis.
        [pixels]     
    '''
    if verbose:
        print('makecat: performing measurements...')
        t0 = time.time()
    
    #Subtract the sky     
    data = data - sky
        
    #Do the measurements
    for src in sources:
        src.add_measurements(half_light( data, segmap, src))
        src.add_measurements(fit_ellipse(data, segmap, src))
        
    #Make the catalogue
    df = pd.concat([s.series for s in sources], axis=1).T
    
    if verbose:
        print('makecat: finished after %i seconds.' % (time.time()-t0))
        t0 = time.time()
    
    return df
    
#==============================================================================
#==============================================================================

