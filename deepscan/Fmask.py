#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:26:38 2017

@author: Dan

"""

import tempfile
import numpy as np
from multiprocessing import Process
from deepscan import NTHREADS, utils

#==============================================================================

class Bounds():
    
    def __init__(self, xmin, ymin, dshape, R):
        self.xmin = int( np.max((0, xmin)) ) 
        self.ymin = int( np.max((0, ymin)) )
        self.xmax = int( np.min((xmin+R, dshape[1])) )
        self.ymax = int( np.min((ymin+R, dshape[0])) )
        self.shape = (self.ymax - self.ymin, self.xmax - self.xmin)


def get_bounds(nthreads, N_per_thread, R, dshape, thread_ID):
    
    '''Get the bounds evenly distributed over the threads'''
    
    #Get minumum coordinates for every thread
    xmins, ymins = np.meshgrid( np.arange(0,dshape[1],step=R), 
                                np.arange(0,dshape[0],step=R) )
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
                          R=R) for xmin, ymin in zip(xmins,ymins)]
    return bounds_list
    
#==============================================================================
    
def mask_region(data, mask, rms, bounds, n_type, sat_level):
    
    '''Modify data array contents by inserting noise into masking region'''
    
    #Get condition to fill region
    cond_replace = mask[bounds.ymin:bounds.ymax, bounds.xmin:bounds.xmax] != 0
                           
    #Get the noise to insert
    if n_type == 'RMS':
            noise = np.random.normal(loc=0, 
                             scale=rms[bounds.ymin:bounds.ymax, 
                                       bounds.xmin:bounds.xmax],
                             size=bounds.shape)
    elif n_type == 'VAR':
        noise =     noise = np.random.normal(loc=0, 
                             scale=np.sqrt(rms[bounds.ymin:bounds.ymax, 
                                               bounds.xmin:bounds.xmax]),
                             size=bounds.shape)
    elif n_type == 'WEIGHT':
        noise = noise = np.random.normal(loc=0, 
                                         scale=1./np.sqrt(rms[bounds.ymin:bounds.ymax, 
                                                              bounds.xmin:bounds.xmax]),
                                         size=bounds.shape)
    else:
        raise ValueError('n_type should be one of: RMS, VAR, WEIGHT')
        
    #Fill in the masked region with noise
    data[bounds.ymin:bounds.ymax,
         bounds.xmin:bounds.xmax][cond_replace] = noise[cond_replace]
    
    #Also set regions above sat_level to zeros
    cond_sat = mask[bounds.ymin:bounds.ymax, bounds.xmin:bounds.xmax]>=sat_level 
    data[bounds.ymin:bounds.ymax,
         bounds.xmin:bounds.xmax][cond_sat] = 0


def mask_regions(data, mask, rms, bounds_list, n_type, sat_level):
    
    '''Appy mask_region over several regions'''
    
    for bounds in bounds_list:
        mask_region(data=data, mask=mask, rms=rms, bounds=bounds, n_type=n_type, sat_level=sat_level)
        
        
def apply_mask(data, mask, noise, R, n_type='RMS', threads=NTHREADS, sat_level=np.inf):
    
    '''Replace all masking regions with noise'''
    
    #Calculate number of R*R regions in data
    Nregions = data.size / (R**2)
    
    #Get the number of regions per thread
    N_per_thread = int(np.floor( Nregions / NTHREADS ))
    
    #Generate the mask in parallel
    pgroup = []
    for n in range(threads):
        
        #Get the boundaires for this thread 
        bounds_list = get_bounds(threads, N_per_thread, R, data.shape, thread_ID=n)
        
        #Run the replacement program
        pgroup.append(Process(target=mask_regions, args=(data, 
                                                         mask,
                                                         noise, 
                                                         bounds_list,
                                                         n_type,
                                                         sat_level)))
        pgroup[-1].start()
    for p in pgroup:
        p.join()
    
#==============================================================================

def fmask(data,
          rms,
          mask,
          R=50, 
          threads=NTHREADS,
          sat_level=np.inf):
    
    #Make a temporary memmap array 
    tfile = tempfile.NamedTemporaryFile(delete=True)
    
    try:
        
        #Create the memmap and fill with data
        data2 = np.memmap(tfile.name, dtype='float32', mode='w+', shape=data.shape)
        data2[:,:] = data[:,:]
            
        #Replace masked pixels with noise        
        apply_mask(data2, mask, rms, R, 'RMS', threads, sat_level=sat_level)
    
        #Save to output array...
        data3 = np.array(data2)
        
    #Remove the temporary file
    finally:
        tfile.close()
        
    return data3


def main(fitsdata, 
         fitsnoise, 
         fitsmask, 
         ofile, 
         R=50, 
         data_ext=0, 
         noise_ext=0, 
         mask_ext=0, 
         n_type = 'RMS',
         threads=NTHREADS,
         sat_level=np.inf):
    
    #Load the data from file
    data = utils.read_fits(fitsdata, extension=data_ext)
    rms = utils.read_fits(fitsnoise, extension=noise_ext)
    mask = utils.read_fits(fitsmask, extension=mask_ext)
    
    #Make a temporary memmap array 
    tfile = tempfile.NamedTemporaryFile(delete=True)

    try:
        
        #Create the memmap and fill with data
        data2 = np.memmap(tfile.name, dtype='float32', mode='w+', shape=data.shape)
        data2[:,:] = data[:,:]
            
        #Replace masked pixels with noise        
        apply_mask(data2, mask, rms, R, n_type, threads, sat_level=sat_level)
    
        #Save to output file
        utils.save_to_fits(data2, fname=ofile, overwrite=True)
        
    #Remove the temporary file
    finally:
        tfile.close()
        

            


