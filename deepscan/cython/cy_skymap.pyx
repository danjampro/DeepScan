#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:47:24 2019

@author: danjampro
"""
import matplotlib.pyplot as plt
import numpy as np

cimport cython
cimport numpy as np
from libcpp cimport bool as cppbool
from libcpp.vector cimport vector as cppvector
from libc.math cimport isnan as cisnan
from cython.operator cimport dereference as deref, preincrement as preinc

#==============================================================================

ctypedef fused data_t:
    np.float32_t
    np.float64_t
    
ctypedef fused mask_t:
    np.uint8_t
    
#==============================================================================

cdef class Mesh():
    
    cdef public:
        Py_ssize_t x, y
        double     sky, rms, sky_, rms_
        double     fillfrac
        tuple slc
        long area
        
    def __cinit__(self, Py_ssize_t x, Py_ssize_t y, tuple slc, 
                  double sky=0, double rms=0, double fillfrac=0):
        
        #Indices on mesh grid
        self.x = x
        self.y = y
        
        self.sky = sky
        self.rms = rms
        
        #Corresponding slices on the image
        self.slc = slc
        self.area = (slc[1].stop-slc[1].start)*(slc[0].stop-slc[0].start)
        
        #Initial values
        self.sky = sky 
        self.rms = rms 
        self.sky_ = 0
        self.rms_ = 0
        
        #Minimum fullness required before interpolation
        self.fillfrac = fillfrac
        
    def plot(self, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot([self.slc[1].start, self.slc[1].stop],
                [self.slc[0].start, self.slc[0].start], 'k-', **kwargs)
        ax.plot([self.slc[1].start, self.slc[1].stop],
                [self.slc[0].stop, self.slc[0].stop], 'k-', **kwargs)
        ax.plot([self.slc[1].stop, self.slc[1].stop],
                [self.slc[0].start, self.slc[0].stop], 'k-', **kwargs)
        ax.plot([self.slc[1].start, self.slc[1].start],
                [self.slc[0].start, self.slc[0].stop], 'k-', **kwargs)
        
#==============================================================================
       
@cython.boundscheck(False)
@cython.wraparound(False)
def measure_sky(data_t[:, :] data, mask_t[:, :] mask, Mesh[:] meshes,
                double fillfrac=0.33):
    '''
    Measure the sky and rms for a group of meshes.  
    '''
    cdef:
        #data_t[:] data_
        Mesh mesh
        Py_ssize_t x, y, idx
        double area, sky_, drms_
        
    if data_t is double:
        dtype = np.float64
    else:
        dtype = np.float32
    
    for mesh in meshes:
                
        area = 0
        idx = 0
        data_ = np.empty(mesh.area, dtype=dtype)
                
        #Get unmasked values
        for x in range(mesh.slc[1].start, mesh.slc[1].stop):
            for y in range(mesh.slc[0].start, mesh.slc[0].stop):
                if mask[y, x] == 0:
                    area += 1
                    data_[idx] = data[y, x]
                    idx += 1
        
        #Ignore if not enough unmasked pixels
        if area / mesh.area < fillfrac:
            mesh.sky = np.nan
            mesh.rms = np.nan
        
        #Apply estimators
        else:       
            sky_, drms_ = np.quantile(data_[:idx], [0.5, 0.159])
            mesh.sky = sky_
            mesh.rms = sky_ - drms_
                        
#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_nans(Mesh[:] meshes, Py_ssize_t ny, Py_ssize_t nx):
    '''
    Fill nan values in between meshes.
    '''
    cdef:
        Py_ssize_t x, y, x_, y_, idx
        Mesh mesh, mesh_
        cppbool finished = False
        cppvector[double] sky_values, rms_values
        double temp
        cppvector[double].iterator it
        
    sky_values.reserve(9)
    rms_values.reserve(9)
        
    while not finished:
        idx = 0
        finished = True  #Terminates when no nans are remaining
                
        #Identify nan meshes
        for y in range(ny):
            for x in range(nx):
                mesh = meshes[idx]
                if cisnan(mesh.sky):
                    sky_values.clear()
                    rms_values.clear()
                    
                    #Find finite neighbours
                    for x_ in range(max(0,x-1), min(x+2, nx)):
                        for y_ in range(max(0,y-1), min(y+2, ny)):
                            mesh_ = meshes[nx*y_ + x_]
                            if not cisnan(mesh_.sky):
                                sky_values.push_back(mesh_.sky)
                                rms_values.push_back(mesh_.rms)
                                
                    #Check if there was a finite neighbour
                    if sky_values.empty():
                        finished = False
                        
                    #Update meshes
                    else:
                        temp = 0
                        it = sky_values.begin()
                        while it != sky_values.end():
                            temp += deref(it)
                            preinc(it)
                        mesh.sky = temp/sky_values.size()
                        
                        temp = 0
                        it = rms_values.begin()
                        while it != rms_values.end():
                            temp += deref(it)
                            preinc(it)
                        mesh.rms = temp/rms_values.size()
                        
                idx += 1 #Next mesh
                        
#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_median_filter(Mesh[:] meshes,Py_ssize_t nx, Py_ssize_t ny, int dmed):
    '''
    Apply median filter to a set of meshes, ignoring nan values.
    '''
    cdef:
        Mesh        mesh
        list        sky, rms
        Py_ssize_t  x_, y_, xmin, xmax, ymin, ymax, idx
        
    for mesh in meshes:
                                
        xmax = min(mesh.x+dmed+1, nx)
        ymax = min(mesh.y+dmed+1, ny)
        xmin = max(mesh.x-dmed, 0)
        ymin = max(mesh.y-dmed, 0)
        
        sky_ = []; rms_ = []
        for x_ in range(xmin, xmax):
            for y_ in range(ymin, ymax):
                idx = nx*y_ + x_
                sky_.append(meshes[idx].sky)
                rms_.append(meshes[idx].rms)
           
        mesh.sky_ = np.nanmedian(sky_)
        mesh.rms_ = np.nanmedian(rms_)
                
#==============================================================================
#==============================================================================

