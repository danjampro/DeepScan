#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:52:40 2017

@author: Dan
"""

import numpy as np

class ellipse():
    
    def __init__(self, a=1, b=1, theta=0, x0=0, y0=0):
        
        self.a = a 
        self.b = b
        self.q = float(b)/a
        self.theta = theta
        self.x0 = x0
        self.y0 = y0
        
    
    def check_inside(self, x, y):
        
        '''Return true if point {x, y} lies within ellipse'''
         
        #Translate ellipse to origin
        x = x - self.x0
        y = y - self.y0
        
        #Rotate ellipse to have semi-major axis aligned with y-axis
        x2 = x*np.cos(-self.theta) - y*np.sin(-self.theta)
        y2 = y*np.cos(-self.theta) + x*np.sin(-self.theta)
        
        A = ( x2/self.b )**2 
        B = ( y2/self.a )**2 
            
        return (A + B) <= 1
        
        
    def draw(self, ax, pts=100, **kwargs):
        
        '''Draw the ellipse on the axis'''
        
        phi = np.linspace(0, 2*np.pi)
        x = self.a * np.cos(phi) 
        y = self.b * np.sin(phi)
        x2 = x*np.cos(self.theta + np.pi/2) - y*np.sin(self.theta + np.pi/2) + self.x0
        y2 = y*np.cos(self.theta + np.pi/2) + x*np.sin(self.theta + np.pi/2) + self.y0
        ax.plot(x2, y2, **kwargs)
        
        
