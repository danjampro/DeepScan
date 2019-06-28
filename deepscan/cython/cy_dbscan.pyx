#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:00:32 2019

@author: danjampro
"""
import numpy as np

cimport cython
cimport numpy as np

#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def get_binarymap(np.uint8_t[:,:] threshed, np.uint8_t[:,:] structure, 
                  long minpts):
    '''
    
    '''
    secarea    = np.zeros_like(threshed, dtype=np.uint8)
    assert(structure.shape[1]%2 != 0)
    assert(structure.shape[0]%2 != 0)
    cdef:
        #np.uint8_t[:, :] threshedv = threshed
        np.uint8_t[:,:]  secareav = secarea
        #np.uint8_t[:,:]  structurev = structure
        Py_ssize_t       nx = threshed.shape[1]
        Py_ssize_t       ny = threshed.shape[0]
        Py_ssize_t       nx_ = structure.shape[1]
        Py_ssize_t       ny_ = structure.shape[0]
        Py_ssize_t       x, y, x_, y_
        Py_ssize_t       dx = (nx_-1)/2
        Py_ssize_t       dy = (ny_-1)/2
        long             total
        
    #Loop over core point candidates
    for x in range(nx):
        for y in range(ny):
            if threshed[y, x] == 1:
                
                total = 0
                for x_ in range(max(0, x-dx), min(nx, x+dx+1)):
                    for y_ in range(max(0, y-dy), min(ny, y+dy+1)):
                        if structure[y_-y+dy, x_-x+dx] == 1:
                            if threshed[y_, x_] > 0:
                                total += 1
                                if total == minpts:
                                    break
                    else:   #If for y_... wasn't broken, continue
                        continue
                                        
                    #Fill the secondary area...
                    for x_ in range(max(0, x-dx), min(nx, x+dx+1)):
                        for y_ in range(max(0, y-dy), min(ny, y+dy+1)):  
                            if structure[y_-y+dy, x_-x+dx] == 1:
                                secareav[y_, x_] = 1
                            
                    #Terminate the loop for this pixel
                    break
            
    return secarea
                        
#==============================================================================
#==============================================================================
