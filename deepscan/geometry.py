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
from astropy.convolution import Tophat2DKernel

#==============================================================================
#Box class

class Box():
    '''
    A Box is a four-sided polygon.
    '''
    def __init__(self, tl, tr, br, bl, label=None):
        '''
        Parameters
        ----------
        tl : tuple of two floats.
            (x, y) coordinate of top left corner.
            
        tr : tuple of two floats.
            (x, y) coordinate of top right corner.
            
        br : tuple of two floats.
            (x, y) coordinate of bottom right corner.
            
        bl : tuple of two floats.
            (x, y) coordinate of bottom left corner.
        '''        
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl
        self.vertices = [tl,tr,br,bl]
        self.label=label
        self.polygon = Polygon([self.tl, self.tr, self.br, self.bl])
    
    
    def draw(self, ax=None, color='k', label=False, **kwargs):
        '''
        Plot the Box's boundary.
        
        Parameters
        ----------
        ax : matplotlib.Axes
            The axes with which to plot the box.
            
        color : str
            The colour of the box.
            
        label : bool
            Add label if True.
            
        **kwargs 
            Passed to matplotlib.pyplot.plot.
            
        Returns
        -------
        matplotlib.Axes
            The plotting axes.
        '''
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
        if label:
            if self.label is not None:
                plt.text(0.5*(self.tl[0]+self.tr[0]),
                         0.5*(self.tl[1]+self.bl[1]),s='{}'.format(self.label),
                         color=color, fontsize=10)
        return ax
                
    
    def check_inside(self, xs, ys):
        '''
        Check if (x,y) coordinate pairs are inside the Box.
        
        Parameters
        ----------
        xs : Iterable of floats
            The x coordinates.
            
        ys : Iterable of floats
            The y coordinates.
        
        Returns
        -------
        1D np.array of bool type
            True if inside.
        '''
        return np.array([self.polygon.contains(Point(p)) for p in zip(xs,ys)])
    
    
    def count_inside(self, xs, ys, maskfrac=0):
        '''
        Count the number of (x,y) coordinate pairs inside the Box.
        
        Parameters
        ----------
        xs : Iterable of floats
            The x coordinates.
            
        ys : Iterable of floats
            The y coordinates.
            
        maskfrac : float between 0 and 1
            The masked fraction with which to correct the count.
                 
        Returns
        -------
        float
            The number of points inside the box.
        '''
        return np.sum(self.check_inside(xs, ys)) * (1.-maskfrac)
    
#==============================================================================
#Ellipse class
        
class Ellipse():
    '''
    A class to handle Ellipses.
    '''
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
        self.area  = np.pi * self.req**2
        
    
    def check_inside(self, x, y):    
        '''
        Return true if point (x, y) lies within ellipse.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''     
        #Translate ellipse to origin
        x = x - self.x0
        y = y - self.y0
                
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) 
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) 
                       
        A = ( y2/self.b )**2 
        B = ( x2/self.a )**2 
            
        return (A + B) <= 1
        
        
    def draw(self, ax=None, color='r', Npoints=100, **kwargs):
        '''
        Plot the ellipse.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        if ax is None:
            ax = plt.gca()
        
        phi = np.linspace(0, 2*np.pi, Npoints)
        x = self.a * np.cos(phi) 
        y = self.b * np.sin(phi)
                        
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) + self.y0
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) + self.x0
        
        ax.plot(x2, y2, color=color, **kwargs)
        
        
    def get_bounds(self, Npoints):
        '''
        Get bounding coordinates of the Ellipse.
        '''
        phi = np.linspace(0, 2*np.pi, Npoints+1)[:-1]
        x = self.a * np.cos(phi) 
        y = self.b * np.sin(phi)
                        
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) + self.y0
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) + self.x0
        return x2, y2        
        
        
    def rescale(self, factor, inplace=False):
        '''
        Resize the Ellipse by factor.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        if inplace:
            self.a *= 2
            self.b *= 2
            return None
        return Ellipse(a=factor*self.a, q=self.q, theta=self.theta, x0=self.x0,
                       y0=self.y0)
        
        
    def get_radii(self, x, y):
        '''
        Get elliptically transformed radii from x, y coordinates.
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        #Translate to origin
        x = x - self.x0
        y = y - self.y0
                
        #Rotate 
        y2 = x*np.cos(-self.theta) - y*np.sin(-self.theta) 
        x2 = y*np.cos(-self.theta) + x*np.sin(-self.theta) 
        
        #Radius estimates
        radii = np.sqrt( ((1./self.a)*x2)**2 + ((1./self.b)*y2)**2 )
        
        return radii
                       
"""       
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
"""

#==============================================================================
#Anulus class
    
class Anulus():
    '''
    
    '''
    def __init__(self, x0, y0, a1, a2, theta=0, q=1):
        '''
        Parameters
        ----------
        
        ''' 
        self.x0 = x0
        self.y0 = y0
        
        eas = [Ellipse(x0=x0,y0=y0,a=a,b=q*a,theta=theta
                                                    ) for a in [a1,a2]]
        self.e1 = eas[np.argmin([a1,a2])]
        self.e2 = eas[np.argmax([a1,a2])]
        
        self.area1 = self.e1.area
        self.area2 = self.e2.area - self.area1
        
        self.area = self.area2 - self.area1
        
        self.Nobjs = None
        
        
    def draw(self, ax=None, color='k', **kwargs):
        '''
        Plot the ellipse.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        if ax is None:
            ax = plt.gca()
        self.e1.draw(color=color, **kwargs)
        self.e2.draw(color=color, **kwargs)
        
        
    def check_inside(self, x, y):
        '''
        Plot the ellipse.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        return (~self.e1.check_inside(x, y)) * self.e2.check_inside(x, y)
        
            
    def count_objects(self, xs, ys):
        '''
        Plot the ellipse.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        self.Nobjs = np.sum(self.check_inside(xs,ys))
        return self.Nobjs
    
#==============================================================================
#Rectangle 
        
class Rectangle():
    '''
    A class to handle Rectangles.
    '''
    def __init__(self, x0=0, y0=0, height=1, width=1):   
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.height = float(height)
        self.width  = float(width)
        
    
    def check_inside(self, x, y):    
        '''
        Return true if point (x, y) lies within Rectangle.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''     
        cond = x > self.x0-(self.width/2)
        cond&= x < self.x0+(self.width/2)
        cond&= y > self.y0-(self.height/2)
        cond&= y < self.y0+(self.height/2)
        return cond
        
        
    def draw(self, ax=None, color='r', **kwargs):
        '''
        Plot the ellipse.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        if ax is None:
            ax = plt.gca()
        ax.plot([self.x0-(self.width/2), self.x0+(self.width/2)],
                [self.y0-(self.height/2), self.y0-(self.height/2)],
                color=color, **kwargs)
        ax.plot([self.x0-(self.width/2), self.x0+(self.width/2)],
                [self.y0+(self.height/2), self.y0+(self.height/2)],
                color=color, **kwargs)
        ax.plot([self.x0-(self.width/2), self.x0-(self.width/2)],
                [self.y0-(self.height/2), self.y0+(self.height/2)],
                color=color, **kwargs)
        ax.plot([self.x0+(self.width/2), self.x0+(self.width/2)],
                [self.y0-(self.height/2), self.y0+(self.height/2)],
                color=color, **kwargs)
        return ax

#==============================================================================
#kernels
        
def unit_tophat(radius):
    '''
    Return a tophat kernel array of unit height.
        
    Parameters
    ----------
    
    Returns
    -------     

    '''
    kernel = Tophat2DKernel(radius).array
    kernel[kernel!=0] = 1
    return kernel

#==============================================================================
#==============================================================================



