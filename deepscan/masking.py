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
    
    shape: shape of output image
    
    ellipses: iterable object containing deepscan.geometry.Ellipse objects
    
    fillval: scalar or iterable containing fill values
    
    dtype: output image data type
    
    Returns
    -------
    
    filled image: image with ellipses filled 
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
    
    







"""

#from . import NTHREADS, BUFFSIZE
#import io, os, tempfile, multiprocessing

def process_init(output_memmap, rms, fillval, buffsize):
    global output_arr, data_arr, rms_arr, fillval_arr, buffsize_
    data_arr = output_memmap
    rms_arr = rms
    fillval_arr = fillval
    buffsize_ = buffsize
    

def mask_ellipses(shape, ellipses, Nthreads=NTHREADS, buffsize=BUFFSIZE,
                  rms=None, fillval=1):
    '''
    Make ellipses.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    with tempfile.TemporaryDirectory() as temppath:
        pool = None
        try:
            
            #Make a memmap array and save to temp file
            data2 = np.memmap(os.path.join(temppath, 'temp.dat'), dtype='float32', mode='w+', shape=data.shape)
            data2[:,:] = data[:,:] #Fill the new array with the data
                                   
            #Generate the mask in parallel
            pool = multiprocessing.Pool(processes=Nthreads, initializer=process_init, initargs=(data2, rms, fillval, buffsize))
            pool.starmap(mask_ellipse_, [[e] for e in ellipses])    
                           
        finally:
            if pool is not None:
                pool.close()
                pool.join()     
    return data2



def mask_ellipse_(ellipse):
    
    '''Mask a single ellipse - memmap'd'''
    
    #Define semi-major axis of masking ellipse
    Rmax = ellipse.a
                   
    #Define rectangulr region to examine
    boxshape = (2*Rmax+1, 2*Rmax+1)
    
    #Min corners of cutout in full data space
    xmin = int(ellipse.x0) - int(boxshape[1]/2)
    ymin = int(ellipse.y0) - int(boxshape[0]/2)
    xmax = xmin + boxshape[1]
    ymax = ymin + boxshape[0]
    
    if (ymax<=0) or (xmax<=0) or (ymin>data_arr.shape[0]) or (
                                                    xmin>data_arr.shape[1]):
        return None
    
    #Select region for masking
    xmin_real = int( np.max((xmin,0)) )
    xmax_real = int( np.min((xmax, data_arr.shape[1])) )
    ymin_real = int( np.max((ymin,0)) )
    ymax_real = int( np.min((ymax, data_arr.shape[0])) )
        
    #Calculate size of array
    nbytes = (ymax_real-ymin_real)*(xmax_real-xmin_real)*data_arr.dtype.itemsize
    if nbytes <= buffsize_:
    
        #Make coordinate grid
        xs, ys = np.meshgrid(np.arange(xmin_real, xmax_real),np.arange(ymin_real, ymax_real))    
        
        #Check if coordinates are inside ellipse
        bools = ellipse.check_inside(xs, ys)
        
        #Filter the coordinates
        xs = xs[bools]
        ys = ys[bools]
            
        #Apply the mask
        if rms_arr is None:
            if (type(fillval_arr) == np.ndarray):
                data_arr[ys,xs] = fillval_arr[ys,xs]
            else:
                data_arr[ys,xs] = fillval_arr
        else:
            if (type(rms_arr) == np.ndarray):
                data_arr[ys,xs] = np.random.normal(loc=0, scale=rms_arr[ys,xs])
            else:
                data_arr[ys,xs] = np.random.normal(loc=0, scale=rms_arr, size=1)        
    
    #Attept to prevent memory error
    else:
        
        #Calculate buffering regions        
        Nperit = int( np.max((1, int( buffsize_ / data_arr.dtype.itemsize) )) )
        xstep = np.min((Nperit, xmax_real))
        ystep = int( buffsize_ / xstep*data_arr.dtype.itemsize ) + 1
        xmins = np.arange(xmin_real, xmax_real, xstep)
        xmaxs = np.arange(xmin_real+xstep, xmax_real+xstep, xstep); xmaxs[-1]=xmax_real
        ymins = np.arange(ymin_real, ymax_real, ystep)
        ymaxs = np.arange(ymin_real+ystep, ymax_real+ystep, ystep); ymaxs[-1]=ymax_real
                    
        for xmin_, xmax_ in zip(xmins, xmaxs):
            for ymin_, ymax_ in zip(ymins, ymaxs):
                                
                #Make coordinate grid
                xs, ys = np.meshgrid(np.arange(xmin_, xmax_),np.arange(ymin_, ymax_))    
        
                #Check if coordinates are inside ellipse
                bools = ellipse.check_inside(xs, ys)
                
                #Filter the coordinates
                xs = xs[bools]
                ys = ys[bools]
                    
                #Apply the mask
                if rms_arr is None:
                    if (type(fillval_arr) == np.ndarray):
                        data_arr[ys,xs] = fillval_arr[ys,xs]
                    else:
                        data_arr[ys,xs] = fillval_arr
                else:
                    if (type(rms_arr) == np.ndarray):
                        data_arr[ys,xs] = np.random.normal(loc=0, scale=rms_arr[ys,xs])
                    else:
                        data_arr[ys,xs] = np.random.normal(loc=0, scale=rms_arr, size=1)   
 

def mask_ellipses2(shape, es, fillvals, dtype='int'):
    mask = np.zeros(shape, dtype=dtype)
    for i, e in enumerate(es):
        Rmax = e.a                       
        boxshape = (2*Rmax+1, 2*Rmax+1)        
        xmin = int(e.x0) - int(boxshape[1]/2)
        ymin = int(e.y0) - int(boxshape[0]/2)
        xmax = xmin + boxshape[1]
        ymax = ymin + boxshape[0]
        xmin = int( np.max((xmin,0)) )
        xmax = int( np.min((xmax, shape[1])) )
        ymin = int( np.max((ymin,0)) )
        ymax = int( np.min((ymax, shape[0])) )
        xs, ys = np.meshgrid(np.arange(xmin, xmax),np.arange(ymin, ymax)) 
        bools = e.check_inside(xs, ys)
        mask[ymin:ymax, xmin:xmax][bools] = fillvals[i] 
    return mask
        
                     
#==============================================================================


class Bounds():
    
    def __init__(self, xmin, ymin, dshape, R):
        self.xmin = int( np.max((0, xmin)) ) 
        self.ymin = int( np.max((0, ymin)) )
        self.xmax = int( np.min((xmin+R, dshape[1])) )
        self.ymax = int( np.min((ymin+R, dshape[0])) )
        self.shape = (self.ymax - self.ymin, self.xmax - self.xmin)


def _get_bounds(nthreads, N_per_thread, boxwidth, dshape, thread_ID):
    
    '''Get the bounds evenly distributed over the threads'''
    
    #Get minumum coordinates for every thread
    xmins, ymins = np.meshgrid( np.arange(0,dshape[1],step=boxwidth), 
                                np.arange(0,dshape[0],step=boxwidth) )
    #Thread specific slice
    idx_low = thread_ID * N_per_thread
    if thread_ID == nthreads-1:
        xmins = xmins.ravel()[ idx_low :]
        ymins = ymins.ravel()[ idx_low :]    
    else:
        idx_high = (thread_ID + 1) * N_per_thread
        xmins = xmins.ravel()[ idx_low : idx_high ]
        ymins = ymins.ravel()[ idx_low : idx_high ]
        
    #Generate bounds for thread
    bounds_list = [Bounds(xmin=xmin,
                          ymin=ymin,
                          dshape=dshape,
                          R=boxwidth) for xmin, ymin in zip(xmins,ymins)]
    return bounds_list
    

    
def _fill_noise(data, mask, sky, rms, bounds_list):
    '''
    Modify data array contents by inserting noise into masking region.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    for bounds in bounds_list:
        
        #Get condition to fill region
        mask_crp = mask[bounds.ymin:bounds.ymax, bounds.xmin:bounds.xmax] 
        
        if mask_crp.any():
            
            #Get the noise to insert
            noise = np.random.normal(loc=0, scale=rms[bounds.ymin:bounds.ymax, 
                                       bounds.xmin:bounds.xmax], size=bounds.shape)
                
            #Fill in the masked region with noise
            data[bounds.ymin:bounds.ymax,
                 bounds.xmin:bounds.xmax][mask_crp] = noise[mask_crp]
            
            #Account for sky background level
            if sky is not None:
                data[bounds.ymin:bounds.ymax,
                     bounds.xmin:bounds.xmax][mask_crp] += sky[mask_crp]
                 

def apply_mask(data, mask, rms=None, sky=None, fillval=1, buffsize=None,
               memmap=True, dtype=None):
               
    if buffsize is None:
        buffsize = int(io.DEFAULT_BUFFER_SIZE / data.dtype.itemsize)
    else:
        buffsize=int(buffsize)
    
    if dtype is None:
        dtype = data.dtype
        
    if sky is None:
        sky = 0
    
    if memmap:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        masked = np.memmap(tfile.name, dtype=dtype, mode='w+', shape=data.shape)
        masked[:,:] = data[:,:]
    else:
        masked = data.copy()
    
    masked = masked.reshape(data.size)
    mask = mask.reshape(data.size)
    
    if rms is None:
        for seg in np.arange(0,data.size+buffsize,buffsize):
            masked[seg:seg+buffsize][mask[seg:seg+buffsize]==1] = fillval
    else:
        rms = rms.reshape(data.size)
        if hasattr(sky, '__len__'):
            for seg in np.arange(0,data.size+buffsize,buffsize):
                masked[seg:seg+buffsize][mask[seg:seg+buffsize]==1
                       ] = np.random.normal(sky[seg:seg+buffsize][mask[
                               seg:seg+buffsize]==1]
                    ,rms[seg:seg+buffsize][mask[seg:seg+buffsize]==1]) 
        else:
            for seg in np.arange(0,data.size+buffsize,buffsize):
                masked[seg:seg+buffsize][mask[seg:seg+buffsize]==1
                       ] = np.random.normal(sky,rms[seg:seg+buffsize][mask[
                               seg:seg+buffsize]==1]) 
    
    return masked.reshape(data.shape)
"""
#==============================================================================