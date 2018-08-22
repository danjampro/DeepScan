#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:19:22 2017

@author: danjampro
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#==============================================================================

class Box():
    
    def __init__(self, tl,tr,br,bl,frame=None):
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl
        self.vertices = [tl,tr,br,bl]
        self.frame=frame
        self.polygon = Polygon([self.tl, self.tr, self.br, self.bl])
    
    def draw(self, ax=None, color='k', frame=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot([self.tl[0],self.tr[0]], [self.tl[1],self.tr[1]],
                                                    color=color, **kwargs)
        ax.plot([self.tr[0],self.br[0]], [self.tr[1],self.br[1]],
                                                    color=color, **kwargs)
        ax.plot([self.br[0],self.bl[0]], [self.br[1],self.bl[1]],
                                                    color=color, **kwargs)
        ax.plot([self.bl[0],self.tl[0]], [self.bl[1],self.tl[1]],
                                                    color=color, **kwargs)
        if frame:
            if self.frame is not None:
                plt.text(self.tl[0], self.tl[1], s=self.frame, color=color,
                         fontsize=5)
                
    def check_inside(self, xs, ys):
        return np.array([self.polygon.contains(Point(p)) for p in zip(xs,ys)])
    
    def count_inside(self, xs, ys, maskfrac=0):
        return np.sum(self.check_inside(xs, ys)) * (1.-maskfrac)
    
    
#==============================================================================

class Ellipse():
    
    def __init__(self, a=1, b=1, theta=0, x0=0, y0=0, q=None):
        
        self.a = abs(a) 
        if q is not None:
            self.b = self.a * q
            self.q = q
        else:
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
                
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) 
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) 
                       
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
                        
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) + self.y0
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) + self.x0
        
        ax.plot(x2, y2, color=color, **kwargs)
        
        
    def rescale(self, factor):
        return Ellipse(a=factor*self.a, q=self.q, theta=self.theta, x0=self.x0, y0=self.y0)
    
    
    
def fit_ellipse(xs,ys,weights=None,rms=False,x0=None,y0=None):
    
    '''
    Fit ellipse to data.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    if weights is None:
        weights = np.ones(len(xs))
        
    #First order moments
    if x0 is None:
        x0 = np.average(xs,weights=weights)
    if y0 is None:
        y0 = np.average(ys,weights=weights)
        
    #Second order moments
    x2 = np.sum( (xs-x0)**2 * weights ) / np.sum(weights)
    y2 = np.sum( (ys-y0)**2 * weights ) / np.sum(weights)
    xy = np.sum( (ys-y0)*(xs-x0) * weights ) / np.sum(weights)
    
    #Handle infinitely thin detections
    if x2*y2 - xy**2 < 1./144:
        x2 += 1./12
        y2 += 1./12
    
    #Calculate position angle
    theta = np.sign(xy) * 0.5*abs( np.arctan2(2*xy, x2-y2) ) + np.pi/2
    
    #Calculate the semimajor & minor axes
    
    c1 = 0.5*(x2+y2)
    c2 = np.sqrt( ((x2-y2)/2)**2 + xy**2 )
    arms = np.sqrt( c1 + c2 )
    brms = np.sqrt( c1 - c2 )

    if not rms:
        dmax = np.sqrt( np.max( ((xs-x0)**2+(ys-y0)**2) ) )
        dmax = np.max((dmax, 1)) #Account for 1-pixel detections
        bmax = (brms/arms)*dmax   
        return Ellipse(x0=x0,y0=y0,a=dmax,b=bmax,theta=theta)

    return Ellipse(x0=x0,y0=y0,a=arms,b=brms,theta=theta)


#==============================================================================


class Anulus():
    def __init__(self, x0, y0, r1, r2, theta=0, q=1):
        eas = [Ellipse(x0=x0,y0=y0,a=r,b=q*r,theta=theta
                                                    ) for r in [r1,r2]]
        self.e1 = eas[np.argmin([r1,r2])]
        self.e2 = eas[np.argmax([r1,r2])]
        
        self.area = np.pi * ((self.e2.a*self.e2.b) - (self.e1.a*self.e1.b))
        
        self.Nobjs = None
        
    def draw(self, ax=None, color='k', **kwargs):
        if ax is None:
            ax = plt.gca()
        self.e1.draw(color=color, **kwargs)
        self.e2.draw(color=color, **kwargs)
        
    def check_inside(self, x, y):
        return (~self.e1.check_inside(x, y)) * self.e2.check_inside(x, y)
        
    def count_objects(self, xs, ys):
        self.Nobjs = np.sum(self.check_inside(xs,ys))
        return self.Nobjs

#==============================================================================
    
def unit_tophat(radius):
    '''Return a tophat kernel array of unit height'''
    from astropy.convolution import Tophat2DKernel
    kernel = Tophat2DKernel(radius).array
    kernel[kernel!=0] = 1
    return kernel


