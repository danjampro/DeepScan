#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:29:33 2019

@author: danjampro
"""
import numpy as np

cimport cython
cimport numpy as np
from libcpp.vector cimport vector as cppvector
from libcpp cimport bool as cppbool
from libcpp.queue cimport queue as cppqueue
from cython.operator cimport dereference as deref, preincrement as preinc

#==============================================================================
#Fused types

#Data
ctypedef fused data_t:
    np.float32_t
    np.float64_t
    
#RMS
ctypedef fused rms_t:
    np.float32_t
    np.float64_t
    
#Bmap
ctypedef fused bmap_t:
    np.uint8_t
    np.int32_t
    np.int64_t
    
#Segmap
ctypedef fused segmap_t:
    np.int32_t
    np.int64_t
    
#Dilation structures
ctypedef fused structure_t:
    np.uint8_t
    
#==============================================================================
      
cdef struct Coordinate:
    Py_ssize_t x, y
            
cdef class Segment:  #See _cy_deblend.pxd
        
    def __cinit__(self, Py_ssize_t xmin, Py_ssize_t xmax, Py_ssize_t ymin,
                  Py_ssize_t ymax, long area, double flux, double fluxmin,
                  double fluxmax, long segID, double rmssum=0,
                  long parentID=0):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.area = area
        self.flux = flux
        self.fluxmin = fluxmin
        self.fluxmax = fluxmax
        self.segID = segID
        self.rmssum = rmssum
        self.parentID = parentID

#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_Label(data_t[:, :] data, bmap_t[:, :] bmap):
    '''
    
    '''
    cdef Py_ssize_t ny = bmap.shape[0]
    cdef Py_ssize_t nx = bmap.shape[1]
    
    #cdef bmap_t[:, :] bmap_view = bmap
    #cdef data_t[:, :] data_view = data
    
    segmap = np.zeros((ny, nx), dtype=np.int32)
    cdef np.int32_t[:, :] segmap_view = segmap
        
    cdef Py_ssize_t x, y, xu, yu, x_, y_, xmin, xmax, ymin, ymax
    cdef np.int32_t segID = 1
    cdef long area
    cdef double flux, fluxmin, fluxmax
    
    cdef list segments = []
        
    cdef Coordinate c, c_    
    cdef cppqueue[Coordinate] toprocess
        
    for y in range(ny):
        for x in range(nx):
            
            #Ignore if labeled
            if segmap_view[y, x] != 0:
                continue
            
            #Check if a cluster should form
            if bmap[y, x] > 0:
                                
                area = 0
                flux = 0
                fluxmin = data[y,x]
                fluxmax = data[y, x]
                xmax = 0; ymax = 0
                ymin = ny; xmin = nx

                c.x = x; c.y = y
                toprocess.push( c )
                
                #Label the whole cluster
                while toprocess.size()!=0:
                                        
                    #Pop a coordinate that needs processing
                    c = toprocess.front()
                    toprocess.pop()
                    xu = c.x; yu = c.y
                                                                                                                            
                    #Queue neighbours
                    for x_ in range(max((xu-1, 0)), min((xu+2, nx))):
                        for y_ in range(max((yu-1, 0)), min((yu+2, ny))):
                            if segmap_view[y_, x_] == 0:
                                if bmap[y_, x_] > 0:
                                    
                                    #Update the segmap
                                    segmap_view[y_, x_] = segID
                                    
                                    #Update the pixel processing list
                                    c_.x = x_
                                    c_.y = y_
                                    toprocess.push( c_ )
                                    
                                    #Update the segment statistics
                                    xmin = min((xmin, x_))
                                    xmax = max((xmax, x_))
                                    ymin = min((ymin, y_))
                                    ymax = max((ymax, y_))
                                    area += 1
                                    fluxmin = min((fluxmin,
                                                   data[y_, x_]))
                                    fluxmax = max((fluxmax,
                                                   data[y_, x_]))
                                    flux += data[y_, x_]
                                        
                #Create a Segment object
                segments.append(Segment(    xmin=xmin, xmax=xmax+1,
                                            ymin=ymin, ymax=ymax+1,
                                            area=area, flux=flux,
                                            segID=segID, fluxmin=fluxmin,
                                            fluxmax=fluxmax) )             
                #Update the label
                segID += 1
                   
    return segmap, segments


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_GetSegments(data_t[:, :] data, segmap_t[:, :] segmap):
    '''
    Make segments from an existing segmap.
    
    This function also works for non-contiguous segments.
    '''
    cdef Py_ssize_t ny = segmap.shape[0]
    cdef Py_ssize_t nx = segmap.shape[1]
    
    #Make an array to mark visited pixels
    visited_ = np.zeros((ny, nx), dtype=np.uint8)
    cdef np.uint8_t[:, :] visited = visited_
            
    cdef Py_ssize_t x, y, xu, yu, x_, y_, xmin, xmax, ymin, ymax
    cdef np.int32_t segID
    cdef long area
    cdef double flux, fluxmin, fluxmax
    
    #cdef list segments = []
    cdef dict segdict = {}
    cdef Segment segment_ 
    cdef int newsegment
        
    cdef Coordinate c, c_    
    cdef cppqueue[Coordinate] toprocess
        
    for y in range(ny):
        for x in range(nx):
            
            #Ignore if not labeled in segmap
            segID = segmap[y, x]
            if segID <= 0:
                continue
            
            #Ignore if already visited
            if visited[y, x] == 1:
                continue
            
            #Else, check if the segment exists already
            if segID in segdict.keys():
                
                newsegment = 0
                                
                #Use segment stats as defaults
                segment_ = segdict[segID]
                area = segment_.area
                flux = segment_.flux
                fluxmin = segment_.fluxmin
                fluxmax = segment_.fluxmax
                xmin = segment_.xmin
                xmax = segment_.xmax
                ymin = segment_.ymin
                ymax = segment_.ymax
                
            else:
                
                newsegment = 1
                      
                #Reset the starting values for the new segment                     
                area = 0
                flux = 0
                fluxmin = data[y,x]
                fluxmax = data[y, x]
                xmax = 0; ymax = 0
                ymin = ny; xmin = nx
    
            c.x = x; c.y = y
            toprocess.push( c )
            
            #Label the whole cluster
            while toprocess.size()!=0:
                                    
                #Pop a coordinate that needs processing
                c = toprocess.front()
                toprocess.pop()
                xu = c.x; yu = c.y
                                                                                                                        
                #Queue neighbours
                for x_ in range(max((xu-1, 0)), min((xu+2, nx))):
                    for y_ in range(max((yu-1, 0)), min((yu+2, ny))):
                        if visited[y_, x_] == 0:
                            if segmap[y_, x_] == segID:
                                
                                #Update the segmap
                                visited[y_, x_] = 1
                                
                                #Update the pixel processing list
                                c_.x = x_
                                c_.y = y_
                                toprocess.push( c_ )
                                
                                #Update the segment statistics
                                xmin = min((xmin, x_))
                                xmax = max((xmax, x_))
                                ymin = min((ymin, y_))
                                ymax = max((ymax, y_))
                                area += 1
                                fluxmin = min((fluxmin,
                                               data[y_, x_]))
                                fluxmax = max((fluxmax,
                                               data[y_, x_]))
                                flux += data[y_, x_]
                                                                    
            #Update the existing segment 
            if newsegment == 0:
                segment_.area = area
                segment_.flux = flux
                segment_.fluxmin = fluxmin
                segment_.fluxmax = fluxmax
                segment_.xmin = xmin
                segment_.xmax = xmax
                segment_.ymin = ymin
                segment_.ymax = ymax     
                
            #OR create a new segment
            else:
                segdict[segID] = Segment(xmin=xmin, xmax=xmax+1,
                                         ymin=ymin, ymax=ymax+1,
                                         area=area, flux=flux,
                                         segID=segID, fluxmin=fluxmin,
                                         fluxmax=fluxmax)
                
    #Return segments sorted by segID
    keys = np.array(list(segdict.keys()))
    keys.sort()
    return [segdict[k] for k in keys]


#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_DeblendSegment(data_t[:, :] data, rms_t[:, :] rms,segmap_t[:, :] segmap,
                      data_t thresh, Segment segment, segmap_t segIDoffset=0,
                      long minarea=5, double contrast = 0.5):
    '''
    
    '''
    #Data views on full array
    cdef segmap_t[:, :] segmap_view = segmap[segment.ymin:segment.ymax,
                                         segment.xmin:segment.xmax]
    cdef data_t[:, :] data_view =  data[segment.ymin:segment.ymax,
                                         segment.xmin:segment.xmax]
    cdef rms_t[:, :] rms_view =   rms[segment.ymin:segment.ymax,
                                         segment.xmin:segment.xmax]
        
    #Mini segmentation map
    cdef Py_ssize_t ny = segment.ymax-segment.ymin
    cdef Py_ssize_t nx = segment.xmax-segment.xmin
    
    #Other variables
    cdef Py_ssize_t x, y, xu, yu, x_, y_, xmin, xmax, ymin, ymax
    cdef segmap_t segIDmax = max(segment.segID, segIDoffset)
    cdef segmap_t segID = segIDmax + 1
    cdef long area
    cdef long areamax = 0
    cdef long segIDroot = 0
    cdef data_t flux, fluxmin, fluxmax
    cdef rms_t rmssum
    cdef list segments = []
    cdef Coordinate c, c_    
    cdef cppqueue[Coordinate] toprocess
            
    #Loop over image
    for x in range(nx):
        for y in range(ny):
            
            #Ignore if incorrect label
            if segmap_view[y, x] != segment.segID: #Unique to this function
                continue
            
            #Check if a cluster should form
            if data_view[y, x] >= thresh:
                                
                area = 0
                flux = 0
                rmssum = 0
                fluxmin = data_view[y, x]
                fluxmax = data_view[y, x]
                xmax = 0; ymax = 0
                ymin = ny; xmin = nx

                c.x = x; c.y = y
                toprocess.push( c )
                
                #Label the whole cluster
                while toprocess.size()!=0:
                                        
                    #Pop a coordinate that needs processing
                    c = toprocess.front()
                    toprocess.pop()
                    xu = c.x; yu = c.y
                                                            
                    #Queue neighbours
                    for x_ in range(max((xu-1, 0)), min((xu+2, nx))):
                        for y_ in range(max((yu-1, 0)), min((yu+2, ny))):
                                if segmap_view[y_, x_] == segment.segID:
                                    if data_view[y_, x_] >= thresh:
                                                                                
                                        #Update the segmap
                                        segmap_view[y_, x_] = segID
                                        
                                        #Update the pixel processing list
                                        c_.x = x_; c_.y = y_
                                        toprocess.push( c_ )
                                        
                                        #Update the segment statistics
                                        xmin = min((xmin, x_))
                                        xmax = max((xmax, x_))
                                        ymin = min((ymin, y_))
                                        ymax = max((ymax, y_))
                                        fluxmin = min((fluxmin,
                                                       data_view[y_, x_]))
                                        fluxmax = max((fluxmax,
                                                       data_view[y_, x_]))
                                        area += 1
                                        flux += data_view[y_, x_]
                                        rmssum += rms_view[y_, x_]
                                        
                #Create a Segment object
                segments.append(  Segment(xmin=xmin+segment.xmin,
                                          xmax=xmax+segment.xmin+1,
                                          ymin=ymin+segment.ymin,
                                          ymax=ymax+segment.ymin+1,
                                          area=area, flux=flux,
                                          segID=segID, rmssum=rmssum,
                                          parentID=segment.segID,
                                          fluxmin=fluxmin, fluxmax=fluxmax) )
                #Find main branch
                if area > areamax:
                    areamax = area
                    segIDroot = segID
                    
                #Update the label
                segID += 1
    
    #Decide whether to keep new segements
    cdef list segIDs_keep = []
    cdef Segment segment_
    for segment_ in segments:
                        
        #Always relabel root segment with original segID
        if segment_.segID == segIDroot:
                                    
            for x in range(segment_.xmin-segment.xmin,
                                           segment_.xmax-segment.xmin):
                for y in range(segment_.ymin-segment.ymin,
                                           segment_.ymax-segment.ymin): 
                    if segmap_view[y, x] == segIDroot:
                        segmap_view[y, x] = segment.segID             
            continue
        
        #Relabel if segment is too small
        if segment_.area < minarea:
            for x in range(segment_.xmin-segment.xmin,
                                           segment_.xmax-segment.xmin):
                for y in range(segment_.ymin-segment.ymin,
                                           segment_.ymax-segment.ymin):  
                    if segmap_view[y, x] == segment_.segID:
                        segmap_view[y, x] = segment.segID
            continue   
        
        #Relabel if too faint
        if segment_.flux-(segment_.area*thresh) < segment_.rmssum * contrast:
            for x in range(segment_.xmin-segment.xmin,
                                           segment_.xmax-segment.xmin):
                for y in range(segment_.ymin-segment.ymin,
                                           segment_.ymax-segment.ymin):  
                    if segmap_view[y, x] == segment_.segID:
                        segmap_view[y, x] = segment.segID
            continue  
        
        segIDs_keep.append(segment_.segID)
        segIDmax = max(segIDmax, segment_.segID)
    
    #These should be only new segments
    segments = [s for s in segments if s.segID in segIDs_keep]        
    return segments, segIDmax

#==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_IndilateSegment(data_t[:, :] data, segmap_t[:, :] segmap,
                       np.uint8_t[:, :] structure, Segment segment):
    '''
    
    '''
    cdef:
        #segmap_t[:, :] segmap_view = segmap
        #data_t[:, :]   data_view =  data   
        #np.uint8_t[:, :] structure_view = structure
        Py_ssize_t ny = data.shape[0]
        Py_ssize_t nx = data.shape[1]
        Py_ssize_t ny_ = structure.shape[0]
        Py_ssize_t nx_ = structure.shape[1]
        Py_ssize_t dx = int(structure.shape[1] / 2)
        Py_ssize_t dy = int(structure.shape[0] / 2)
        
        cppvector[Coordinate] coordinates      
        Coordinate c
        Py_ssize_t x, x_
        Py_ssize_t y, y_
             
    coordinates.reserve(segment.area)
    
    #Get coordinates for dilation
    for x in range(segment.xmin, segment.xmax):
        for y in range(segment.ymin, segment.ymax):
            if segmap[y, x] == segment.segID:
                coordinates.push_back(Coordinate(x=x, y=y))
                    
    #Perform dilation using c++ iterator
    cdef cppvector[Coordinate].iterator it = coordinates.begin()
    while it != coordinates.end():
        c = deref(it)
        x = c.x; y = c.y
        for x_ in range(max((0, x-dx)), min((x+dx+1, nx))):
            for y_ in range(max((0, y-dy)), min((y+dy+1, ny))):
                if structure[y_-y+dy, x_-x+dx] != 0:
                    if segmap[y_, x_] == segment.parentID:
                        if data[y_, x_] < segment.fluxmin:
                            segmap[y_, x_] = segment.segID
                            segment.xmin = min(segment.xmin,x_)
                            segment.xmax = max(segment.xmax,x_)
                            segment.ymin = min(segment.ymin,y_)
                            segment.ymax = max(segment.ymax,y_)
                            segment.area+= 1
        #Increment the iterator
        preinc(it)
                   
#==============================================================================
#==============================================================================



    
        

                
            
    
    
