#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:48:49 2017

@author: danjampro
"""

import numpy as np
from . import geometry, SB

def fit_ellipse(xs,ys,weights=None,rms=False):
    
    #First order moments
    x0 = np.average(xs,weights=weights)
    y0 = np.average(ys,weights=weights)
    
    #Second order moments
    x2 = np.average(xs**2,weights=weights) - x0**2
    y2 = np.average(ys**2,weights=weights) - y0**2
    xy = np.average(xs*ys,weights=weights) - x0*y0
    
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
        return geometry.ellipse(x0=x0,y0=y0,a=dmax,b=bmax,theta=theta)

    return geometry.ellipse(x0=x0,y0=y0,a=arms,b=brms,theta=theta)


class Source():
    
    def __init__(self, label, cslice):
        self.ellipse_max = None
        self.ellipse_rms = None
        self.ellipse_rms_weighted = None
        self.xs = None
        self.ys = None
        self.Is = None
        self.cslice = cslice
        self.label = label
    
    def get_crds(self,clusters, mask=None):
        if ((self.xs is None)*(self.ys is None)):
            
            xs, ys = np.meshgrid(np.arange(self.cslice[1].start,self.cslice[1].stop),
                                 np.arange(self.cslice[0].start,self.cslice[0].stop))
            cond = clusters[self.cslice]==self.label
            
            #Mask condition
            if mask is not None:
                cond *= mask[self.cslice] == 0
            
            self.xs = xs[cond]
            self.ys = ys[cond]
        return self.xs, self.ys
    
    def get_data(self, data, clusters, mask=None):
        xs, ys = self.get_crds(clusters, mask=mask)
        if self.Is is None:
            self.Is = data[ys, xs]
        return self.Is
    
    def get_ellipse_max(self, segmap, mask=None):
        xs, ys = self.get_crds(segmap, mask=mask)
        self.ellipse_max=fit_ellipse(xs,ys,weights=None,rms=False)
        return self.ellipse_max
    
    def get_ellipse_rms(self, clusters, mask=None):
        xs, ys = self.get_crds(clusters, mask=mask)
        self.ellipse_rms=fit_ellipse(self.xs,self.ys,weights=None,rms=True)
        return self.ellipse_rms
    
    def get_ellipse_rms_weighted(self, data, segmap, mask=None):
        Is = self.get_data(data, segmap, mask=mask) #xs & ys are set in get_data
        self.ellipse_max_weighted=fit_ellipse(self.xs,self.ys,weights=Is,rms=True)
        return self.ellipse_max_weighted
    
    
    
    def fit_1Dsersic(self, data, segmap, uiso, ps, mzero, dr=5, Rmax=250, mask=None, **kwargs):
        
        from scipy.optimize import curve_fit
        from . import sersic
        
        #Get weighted ellipse
        e_weight = self.get_ellipse_rms_weighted(data, segmap, **kwargs)
        
        #Get data cutout
        cutout = data[int(e_weight.y0-Rmax):int(e_weight.y0+Rmax),
                      int(e_weight.x0-Rmax):int(e_weight.x0+Rmax)]
        
        #Get mask cutout
        if mask is not None:
            mask_crp = mask[int(e_weight.y0-Rmax):int(e_weight.y0+Rmax),
                          int(e_weight.x0-Rmax):int(e_weight.x0+Rmax)]
        else:
            mask_crp = False
        
        #Define coordinate grid
        xx, yy = np.meshgrid(np.arange(cutout.shape[1])-Rmax, np.arange(cutout.shape[0])-Rmax)
        x0 = Rmax; y0=Rmax
        
        r = 0
        Icrit = SB.SB2Counts(uiso, ps, mzero)
        Is = []
        dIs = []
        rs = []
        while True:
            
            #Get inside ellipse condition
            e1 = geometry.ellipse(x0=x0, y0=y0, a=r, b=r*e_weight.q, theta=e_weight.theta)
            e2 = geometry.ellipse(x0=x0, y0=y0, a=r+dr, b=(r+dr)*e_weight.q, theta=e_weight.theta)
        
            inside = e2.check_inside(xx,yy) * ~e1.check_inside(xx,yy) * ~mask_crp
            
            I = np.median(cutout[inside])
            dI = np.std(cutout[inside]) / np.sqrt(inside.sum())
            
            if np.isfinite(I) * (inside.sum() > 2):
                Is.append(I)
                dIs.append(dI)
                rs.append(0.5*(2*r + dr))
                
                #Break condition
                if I <= Icrit:
                    break
            #Increase the radius    
            r += dr
    
            #Max radius condition
            if r >= Rmax:
                break
        
        try:
            popt, pcov = curve_fit(sersic.profile, xdata=rs, ydata=Is, sigma=dIs)
            return popt
        except:
            return None
            
        
            
        
        
        
        
        
        
    
    
    def display(self, data, ax=None, mapping=np.arcsinh, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(mapping(data[self.cslice]), **kwargs)
        