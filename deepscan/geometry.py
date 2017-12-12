#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:19:22 2017

@author: danjampro
"""

import numpy as np
import matplotlib.pyplot as plt

#==============================================================================

class box():
    def __init__(self, xmin, xmax, ymin, ymax,x0=None,y0=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        if x0 is None:
            x0 = (xmax + xmin) / 2
        if y0 is None:
            y0 = (ymax + ymin) / 2
        self.x0 = x0
        self.y0 = y0
            
    def fill(self, data, fillval=1):
        data[self.ymin:self.ymax,self.xmin:self.xmax] = fillval
        return data
    
    def draw(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot([self.xmin, self.xmax], [self.ymin, self.ymin], 'b-')
        ax.plot([self.xmin, self.xmax], [self.ymax, self.ymax], 'b-')
        ax.plot([self.xmin, self.xmin], [self.ymin, self.ymax], 'b-')
        ax.plot([self.xmax, self.xmax], [self.ymin, self.ymax], 'b-')
        
    def check_inside(self, x, y):
        return (x < self.xmax) * (x >= self.xmin) * (y < self.ymax) * (y >= self.ymin)
  
    
#==============================================================================
'''
class rectangle():
    
    def __init__(self, x0, y0, theta, width, height):
        self.x0 = x0
        self.y0 = y0
        self.theta = theta
        self.width = width
        self.height = height
        self.slc = (slice(int(np.floor(-0.5*height*np.sin(theta)+y0)),
                          int(np.ceil(0.5*height*np.sin(theta)+y0))),
                    slice(int(np.floor(-0.5*width*np.cos(theta)+x0)),
                          int(np.ceil(0.5*width*np.cos(theta)+x0))))
        self.m1a = 
        
    def fill(self, data, fillval=1):
        cond = np.zeros((self.slc[0].stop-self.slc[0].start,
                         self.slc[1].stop-self.slc[1].start), dtype='bool')
        
        data[self.ymin:self.ymax,self.xmin:self.xmax] = fillval
        return data
    
    def draw(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot([self.xmin, self.xmax], [self.ymin, self.ymin], 'b-')
        ax.plot([self.xmin, self.xmax], [self.ymax, self.ymax], 'b-')
        ax.plot([self.xmin, self.xmin], [self.ymin, self.ymax], 'b-')
        ax.plot([self.xmax, self.xmax], [self.ymin, self.ymax], 'b-')
        
    def check_inside(self, x, y):
        return (x < self.xmax) * (x >= self.xmin) * (y < self.ymax) * (y >= self.ymin)
'''
#==============================================================================

class ellipse():
    
    def __init__(self, a=1, b=1, theta=0, x0=0, y0=0):
        
        self.a = abs(a) 
        self.b = abs(b)
        self.q = float(b)/a
        self.theta = theta
        self.x0 = x0
        self.y0 = y0
        self.req = np.sqrt(self.a*self.b)
        
    
    def check_inside(self, x, y):
        
        '''Return true if point {x, y} lies within ellipse'''
         
        #Translate ellipse to origin
        x = x - self.x0
        y = y - self.y0
        
        #Rotate ellipse to have semi-major axis aligned with y-axis
        #y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta)
        #x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta)
        
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) 
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) 
               
        #A = ( x2/self.b )**2 
        #B = ( y2/self.a )**2 
        
        A = ( y2/self.b )**2 
        B = ( x2/self.a )**2 
            
        return (A + B) <= 1
        
        
    def draw(self, ax=None, color='r', pts=100, **kwargs):
        
        '''Draw the ellipse on the axis'''
        if ax is None:
            ax = plt.gca()
        
        phi = np.linspace(0, 2*np.pi)
        x = self.a * np.cos(phi) 
        y = self.b * np.sin(phi)
                
        #x2 = x*np.cos(self.theta) - y*np.sin(self.theta) + self.x0
        #y2 = y*np.cos(self.theta) + x*np.sin(self.theta) + self.y0
        
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) + self.y0
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) + self.x0
        
        ax.plot(x2, y2, color=color, **kwargs)
        
        
    def rescale(self, factor):
        return ellipse(a=factor*self.a, b=factor*self.b, theta=self.theta, x0=self.x0, y0=self.y0)
    
#==============================================================================

def unit_tophat(radius):
    '''Return a tophat kernel array of unit height'''
    from astropy.convolution import Tophat2DKernel
    kernel = Tophat2DKernel(radius).array
    kernel[kernel!=0] = 1
    return kernel