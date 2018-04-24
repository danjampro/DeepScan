#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:18:26 2017

@author: danjampro

"""

import time
from deepscan import masking, geometry, NTHREADS, convolution
import numpy as np
from scipy.ndimage.measurements import label, maximum_position
#from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial


#"""
def find_sat_regions(data, saturate):
    '''
    Find the unique saturated regions in the data.
    
    Parameters
    ----------
    
    data (2d float array): data array.
    
    saturate (float): saturation level [ADU]
    
    Returns
    -------
    
    float, 2d float array: Number of objects and object mask
    '''
    labeled, Nobjs = label(data>saturate)
    return Nobjs, labeled



def dilate_large(mask, dilation_kernel, tol=1E-5, **kwargs):
    '''
    Perform binary dilation on mask using FFT convolution.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    conv = convolution.convolve_large(mask, kernel=dilation_kernel, **kwargs)
    conv[conv<=tol] = 0
    conv[conv!=0] = 1
    return conv



def estimator_sigclip(data, sigma=3):
    from astropy.stats import sigma_clip
    return np.mean(sigma_clip(data, sigma))
    
def measure_average_flux(data, x0, y0, r0, dr, mask=None,
                                                 estimator=estimator_sigclip):
    '''
    Calculate the average flux within an annulus.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    #Calculate bounds of minimum bounding rectangle
    xmin = int(np.max((x0-r0-dr, 0)))
    xmax = int(np.min((x0+r0+dr+1, data.shape[1])))
    ymin = int(np.max((y0-r0-dr, 0)))
    ymax = int(np.min((y0+r0+dr+1, data.shape[0])))
    
    #Crop the mask
    if mask is None:
        mask_crop=~np.isfinite(data[ymin:ymax, xmin:xmax])
    else:
        mask_crop=mask[ymin:ymax, xmin:xmax]
        
    #Create the annulus
    xx, yy = np.meshgrid(np.arange(xmin, xmax),np.arange(ymin, ymax))    
    xx -= x0
    yy -= y0    
    d2s = xx**2 + yy**2
    cond = (d2s >= r0**2) * (d2s < (r0+dr)**2) * ~mask_crop
    
    if cond.any():
        #Use the estimator to calculate an average flux value
        return estimator(data[ymin:ymax, xmin:xmax][cond])
    else:
        return np.inf
    


def _fit_apertures(mps, Icrit, Rmax, dr):
    
    apertures = []
    
    #Loop over objects in chunk
    for mp in mps:
        mp = int(mp[0]), int(mp[1])
                
        #Set up while loop
        i=0
        while True:
            
            #Save the radius
            r = (i+1)*dr
                    
            #Measure the average SB within annulus
            sb = measure_average_flux(data_ap, mp[1], mp[0], i*dr, dr, mask=mask_ap) 
            
            #If average SB is below threshold then break and save ellipse
            if sb < Icrit: 
                aperture = geometry.ellipse(x0=mp[1],y0=mp[0],a=r,b=r)
                break
            
            #Break the loop if maximum radius is reached
            elif r >= Rmax:
                aperture=geometry.ellipse(x0=mp[1],y0=mp[0],a=Rmax,b=Rmax)
                print('WARNING: Starmask aperture at (%i, %i) has reached maximum size.' % (mp[0], mp[1]))
                break
            
            #Update counter
            i+=1
            
        apertures.append(aperture)
        
    return apertures
           
    

def _init_fit_apertures(data, mask):
    '''
    Initializer for fit_apertures process pool.

    Parameters
    ----------

    Results
    -------

    '''
    global data_ap, mask_ap
    data_ap = data   
    mask_ap = mask
    
def fit_apertures(data, Icrit, Nobjs, labeled, convolved=None, mask=None, dr=5,
                  Rmax=1E+4, Nthreads=NTHREADS):
    '''
    Fit circular apertures to saturated regions.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''     
    if convolved is None:
        convolved = data
    
    #Find location of maximum in convolved image
    #This helps find the true centre of the saturated regions
    mps =  maximum_position(convolved, labels=labeled, index=np.arange(1,Nobjs+1))
        
    #Do calculations for parallelisation
    n = int(len(mps)/Nthreads)
    chunks = [mps[i:i + n] for i in range(0, len(mps), n)]
            
    #Create the pool
    pool = Pool(Nthreads, initializer=_init_fit_apertures, initargs=[data, mask])
    try:
        apertures = pool.map(partial(_fit_apertures, Icrit=Icrit, Rmax=Rmax,
                                     dr=dr), chunks)
    finally:
        pool.close()
        pool.join()
            
    #Flatten list
    apertures_ = []
    for aps in apertures:
        apertures_.extend(aps)
     
    return apertures_
        


def starmask(data, saturate, Icrit, convolve_size=25, dilate_size=15, dr=20,
             Rmax=1000, makeplots=False, verbose=True, Nthreads=NTHREADS,
             mask=False, **kwargs):
    '''
    Create a starmask.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    t0 = time.time()
    if verbose:
        print('starmask: finding objects...')
        
    #Find the saturated regions in the original image
    Nobjs, labeled = find_sat_regions(data, saturate)
    
    #Make a convolved image if necessary
    if convolve_size:
        convolve_kernel = geometry.unit_tophat(convolve_size)
        convolve_kernel /= convolve_kernel.sum()
        convolved = convolution.convolve_large(data, convolve_kernel)
    else:
        convolved = None
        
    #Dilate the saturation regions if necessary
    if dilate_size:
        dilate_kernel = geometry.unit_tophat(dilate_size)
        dilated = dilate_large((labeled!=0).astype('int'), dilate_kernel).astype('bool')
    else:
        dilated = (labeled!=0).astype('bool')
        
    if verbose:
        print('starmask: fitting apertures...')
        
    #Fit the apertures around the saturation regions
    #dilated is used as a mask for the average flux calculation
    aps = fit_apertures(data, Icrit, Nobjs, labeled, convolved=convolved, dr=dr,
                        Rmax=Rmax, mask=dilated+mask, Nthreads=NTHREADS)
            
    #Create the output mask
    mask_ = masking.mask_ellipses(ellipses=aps, data=np.zeros_like(data, dtype='bool'),
                         rms=None, fillval=True, **kwargs).astype('bool')
    mask_ += dilated
    '''
    mask += dilated
    mask += mask_
    '''
    
    t1 = time.time() - t0
    
    #Create diagnostic plot
    if makeplots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.arcsinh(data), cmap='binary')
        plt.contour(dilated, colors='r')
        [e.draw(color='b') for e in aps]
        [geometry.ellipse(x0=e.x0,y0=e.y0,a=e.a-dr,b=e.b-dr,theta=0).draw(color='b',
                                                 linestyle='--') for e in aps if e.a!=dr]
        [plt.plot(e.x0, e.y0, 'b+') for e in aps]
        plt.xlim(0, data.shape[1])
        plt.ylim(data.shape[0], 0)
        
    print('starmask: finished after %i seconds.' % t1)
    '''
    return mask
    '''
    return mask_


"""    

from scipy.ndimage.filters import median_filter, maximum_filter
from scipy.ndimage.measurements import maximum
import os, tempfile, multiprocessing, shutil


def _apply_init(output, memmap):
    global image_, applied_
    image_ = memmap
    applied_ = output
    
def _apply(slc, func, overlap, shape_idx, kernel=None, **kwargs):

    if shape_idx==1:
        overlap = np.max((0, -slc[1].stop-overlap+image_.shape[1]))
        slc2 = (slc[0], slice(slc[1].start, slc[1].stop+overlap))
        applied_[slc] = func(image_[slc2], **kwargs)
        if overlap == 0:
            applied_[slc] = func(image_[slc2], **kwargs)
        else:
            applied_[slc] = func(image_[slc2], **kwargs)[:,:-overlap]
    else:
        overlap = np.max((0, -slc[0].stop-overlap+image_.shape[0]))
        slc2 = (slice(slc[0].start, slc[0].stop+overlap), slc[1])
        if overlap == 0:
            applied_[slc] = func(image_[slc2], **kwargs)
        else:
            applied_[slc] = func(image_[slc2], **kwargs)[:-overlap,:]


def apply(image, func, kernel=None, overlap=0, Nthreads=NTHREADS, **kwargs):
            
    tempdir = tempfile.mkdtemp()
    pool = None
    
    try:      
        output = np.memmap(os.path.join(tempdir, 'applied.dat'),
                       shape=image.shape, dtype=image.dtype, mode='w+')
        
        pool = multiprocessing.Pool(Nthreads, initializer=_apply_init,
                                    initargs=(output,image))
        
        shape_idx = np.argmax(image.shape)
        
        Nperthread = int(np.ceil(image.shape[shape_idx] / Nthreads))
        
        cmins = np.arange(0, image.shape[shape_idx], Nperthread)
        cmaxs = cmins + Nperthread; cmaxs[-1] = image.shape[shape_idx]
        
        if shape_idx == 0:
            slices = [(slice(cmins[i],cmaxs[i]), slice(0,image.shape[1])) for i in range(cmins.size)]
        else:
            slices = [(slice(0,image.shape[0]),slice(cmins[i],cmaxs[i])) for i in range(cmins.size)]
        
        pfunc = partial(_apply, func=func, kernel=kernel,
                        overlap=overlap, shape_idx=shape_idx,
                        **kwargs)
        
        pool.map(pfunc, slices)
        
    finally:
        shutil.rmtree(tempdir)
        if pool is not None:
            pool.close()
            pool.join()
        
    return output
        
    

def starmask(data, rms, saturate, eps, thresh, kappa, ps, medfilt_size=15,
             maxfilt_size=25, Nthreads=NTHREADS, verbose=False, debug=False):
    
    if verbose: print('starmask: creating mask...')
    t0 = time.time()
    
    smooth = apply(data, median_filter, size=medfilt_size, overlap=medfilt_size,
                   Nthreads=Nthreads)
    
    ta = time.time() - t0
    
    print(ta)
    
    mx = apply(data, maximum_filter, size=maxfilt_size, overlap=maxfilt_size,
               Nthreads=Nthreads)
    
    tb = time.time() - ta - t0
    
    print(tb)

    C = dbscan.dbscan(data, eps=eps, ps=ps, kappa=kappa, rms=rms, thresh=thresh, lowmem=True,
                      verbose=verbose)
    
    maxs = maximum(data, labels=C.segmap_dilate, index=np.arange(C.Nobjs)+1)
    
    bsrcs = [src for i, src in enumerate(C.sources) if maxs[i] >= saturate]
    
    es = [src.get_ellipse_max(C.segmap_dilate) for src in bsrcs]
    
    mask = mx >= saturate
    
    mask += masking.mask_ellipses(np.zeros_like(data, dtype='bool'), es,
                                  rms=None, fillval=1).astype('bool')
    
    t1 = time.time() - t0
    if verbose: print('starmask: finished after %i seconds' % t1)
    
    if debug:
        return mask, C
    
    return mask
"""
