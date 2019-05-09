#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:57:35 2017

@author: danjampro
"""
import numpy as np

#==============================================================================

def fill_ellipses(shape, ellipses, fillval=1, dtype='bool'):
    '''
    Create 2D array of filled ellipses.
    
    Parameters
    ----------
    shape : tuple of ints 
        shape of output image.
    
    ellipses : list of deepscan.geometry.Ellipse objects
        The ellipses to fill.
    
    fillval : float or 2D array of shape
        Value(s) to fill in the ellipses.
    
    dtype : type
        Type of output image.
    
    Returns
    -------  
    2D array of shape 
        Filled image.
    '''
    mask = np.zeros(shape, dtype=dtype)
    
    for i, e in enumerate(ellipses):
        
        #Define semi-major axis of masking ellipse
        Rmax = e.a
                       
        #Define rectangulr region to examine
        boxshape = (2*Rmax+1, 2*Rmax+1)
        
        #Min corners of cutout 
        xmin = int(e.x0) - int(boxshape[1]/2)
        ymin = int(e.y0) - int(boxshape[0]/2)
        xmax = xmin + boxshape[1]
        ymax = ymin + boxshape[0]
        
        #Crop at image borders
        xmin = int( np.max((xmin,0)) )
        xmax = int( np.min((xmax, shape[1])) )
        ymin = int( np.max((ymin,0)) )
        ymax = int(np.min((ymax, shape[0])))
        
        if ((xmax<=0) or (ymax<=0) or (xmin>shape[1]) or (ymin>shape[0])):
            continue
        
        xx, yy = np.meshgrid(np.arange(xmin,xmax), np.arange(ymin,ymax))

        cond = e.check_inside(xx, yy)
        
        if hasattr(fillval, '__len__'):
            mask[yy[cond], xx[cond]] = fillval[i]
        else:
            mask[yy[cond], xx[cond]] = fillval
        
    return mask


def apply_mask(data, mask, rms=None, sky=None, copy=True, fillval=0):
    '''
    Apply the mask to the data.
    
    Parameters
    ----------
    
    data: Input data np.array.
    
    mask: Input mask np.array.
    
    rms (optional): Will fill mask with noise if this is provided (np.array).
    
    sky (optional): Will add this to noise if rms is provided (np.array).
    
    copy: Copy data array?
    
    fillval: Scalar fill value used if rms is not specified.
    
    Returns
    -------
    data: Modified np.array.
    '''
    mask = mask.astype('bool')
    if copy:
        data = data.copy()
    if sky is None:
        sky = 0
    if rms is not None:
        if hasattr(sky, '__len__'):
            data[mask] = np.random.normal(sky[mask], rms[mask])
        else:
            data[mask] = np.random.normal(sky, rms[mask])
    else:
        data[mask] = fillval
    return data
        
#==============================================================================
#==============================================================================

    







