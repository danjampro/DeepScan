#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:04:55 2017

@author: Dan

Source masking using DBSCAN
"""

import numpy as np
from . import NTHREADS, BUFFSIZE
import os, tempfile, shutil, multiprocessing

#==============================================================================

def process_init(output_memmap, rms, fillval, buffsize):
    global output_arr, data_arr, rms_arr, fillval_arr, buffsize_
    data_arr = output_memmap
    rms_arr = rms
    fillval_arr = fillval
    buffsize_ = buffsize
def source_mask(data, ellipses, noise, Nprocs=NTHREADS, buffsize=BUFFSIZE, fillval=0):
    
    '''Make mask based on cluster file'''

    temppath = tempfile.mkdtemp()
    pool = None
    try:
        
        #Make a memmap array and save to temp file
        data2 = np.memmap(os.path.join(temppath, 'temp.dat'), dtype='float32', mode='w+', shape=data.shape)
        data2[:,:] = data[:,:] #Fill the new array with the data
                               
        #Generate the mask in parallel
        pool = multiprocessing.Pool(processes=Nprocs, initializer=process_init, initargs=(data2, noise, fillval, buffsize))
        pool.starmap(mask_ellipse, [[e] for e in ellipses])    
               
        #Save the memmap array
        result = np.array(data2)
        
    finally:
        shutil.rmtree(temppath)
        if pool is not None:
            pool.close()
            pool.join()
        del data2
    
    return result



def mask_ellipse(ellipse):
    
    '''Mask a single ellipse'''
           
    #Define semi-major axis of masking ellipse
    Rmax = ellipse.a
                   
    #Define rectangulr region to examine
    boxshape = (2*Rmax+1, 2*Rmax+1)
    
    #Min corners of cutout in full data space
    xmin = int(ellipse.x0) - int(boxshape[1]/2)
    ymin = int(ellipse.y0) - int(boxshape[0]/2)
    xmax = xmin + boxshape[1]
    ymax = ymin + boxshape[0]
    
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

            
#==============================================================================

def main(ofile, fitsdata, fitsnoise, cfile, Nprocs=NTHREADS, overwrite=True, ext_data=0, ext_noise=0, n_type='rms'):
    
    '''Mask the clusters in the original data and save the output as fits'''
    
    from . import utils, dbscan
    
    #Read data and noise
    data = utils.read_fits(fitsdata, extension=ext_data)
    noise = utils.read_fits(fitsnoise, extension=ext_noise)
    
    if n_type == 'var':
        noise = np.sqrt(noise)
    elif n_type == 'weight':
        noise = 1./np.sqrt(noise)
    else:
        assert(n_type=='rms')
    
    #Read clusters as ellipses
    Es = dbscan.read_ellipses(cfile)
    
    #Do the masking
    data2 = source_mask(data, Es, noise, Nprocs=Nprocs)
    
    #Save as fits
    utils.save_to_fits(data2, ofile, overwrite=overwrite)
    

    
    
    
    



