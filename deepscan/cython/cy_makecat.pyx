#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:00:32 2019

@author: danjampro
"""
import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport sqrt as csqrt 
from libc.math cimport atan2 as cpparctan2
from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector as cppvector
from libcpp cimport bool as cppbool
from libcpp.queue cimport queue as cppqueue
from cython.operator cimport dereference as deref, preincrement as preinc

from .cy_deblend cimport Segment
from .cy_deblend import Segment

#==============================================================================
#Typedefs

ctypedef fused data_t:
    np.float32_t
    np.float64_t    

#Segmap
ctypedef fused segmap_t:
    np.int32_t
    np.int64_t
        
#==============================================================================
#Structs and classes
    
cpdef Segment identity(Segment obj):
    return obj
    
cdef struct Pixel:
    Py_ssize_t x, y
    double value
        
cdef cppbool Pixel_greater(Pixel &a, Pixel&b):
    return a.value > b.value
        
#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def measure_segment(data_t[:,:] data, segmap_t[:,:] segmap,
                    Segment segment):
    '''
    
    '''
    cdef:
        cppvector[Pixel] pixels
        Py_ssize_t nx, ny, x, y
        segmap_t segID = segment.segID
        data_t I50, I50av
        double R50
        double flux = 0
        long area = 0
        double pi = np.pi
        
    pixels.reserve(segment.area)
    
    #Get the data for the segment
    nx = data.shape[1]; ny = data.shape[0]
    for x in range(segment.xmin, segment.xmax):
        for y in range(segment.ymin, segment.ymax):
            if segmap[y, x] == segID:
                pixels.push_back(Pixel(x=x, y=y, value=data[y, x]))
                area += 1
                flux += data[y, x]
                
    #Sort the pixels in decending order
    stdsort(pixels.begin(), pixels.end(), Pixel_greater)
    
    #Estimate the half-light properties
    cdef data_t halfflux = flux / 2
    cdef data_t cumflux = 0
    cdef long N50 = 0
    
    cdef cppvector[Pixel].iterator it = pixels.begin()
    while it != pixels.end():
        
        I50 = deref(it).value
        cumflux += I50
        N50 += 1
        if cumflux > halfflux:
            break        
        preinc(it)
        
    R50 = csqrt(N50 / pi)
    I50av = cumflux / N50
    
    #Estimate the elliptical parameters
    cdef:
        double x0, y0, arms, brms, q, theta, c1, c2
        double numx = 0
        double numy = 0
        double x2 = 0
        double y2 = 0
        double xy = 0
        Pixel pixel
        
    #First order moments
    it = pixels.begin()
    while it != pixels.end(): 
        pixel = deref(it)
        numx += pixel.value * pixel.x
        numy += pixel.value * pixel.y
        preinc(it)
    x0 = numx / flux; y0 = numy / flux
        
    #Second order moments
    it = pixels.begin()
    while it != pixels.end(): 
        pixel = deref(it)
        x2 += (pixel.x-x0)**2 * pixel.value
        y2 += (pixel.y-y0)**2 * pixel.value
        xy += (pixel.x-x0)*(pixel.y-y0) * pixel.value
        preinc(it)
    x2 /= flux; y2 /= flux; xy /= flux
    
    #Handle infinitely thin detections 
    if x2*y2 - xy**2 < 1./144:
        x2 += 1./12
        y2 += 1./12
    
    #Position angle 
    theta = pi/2 + 0.5*abs( cpparctan2(2*xy, x2-y2) ) # *np.sign(xy) 
    if xy < 0:
        theta *= -1
    
    #Semimajor & minor axes  
    c1 = 0.5*(x2+y2)
    c2 = csqrt( ((x2-y2)/2)**2 + xy**2 )
    arms = csqrt( c1 + c2 )
    brms = csqrt( c1 - c2 )
    q = brms / arms
    
    #Return as Python dict
    return {'a_rms':arms, 'b_rms':brms, 'theta':theta, 'q':q, 'xcen':x0,
            'ycen':y0, 'segID':segment.segID, 'flux':flux, 'area':area,
            'I50':I50, 'I50av':I50av, 'R50':R50, 'parentID':segment.parentID}
        
#==============================================================================
#==============================================================================

        
    
        