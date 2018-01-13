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
    
    def get_crds(self, clusters, mask=None):
        if ((self.xs is None)*(self.ys is None)):
            
            xs, ys = np.meshgrid(np.arange(self.cslice[1].start,self.cslice[1].stop),
                                 np.arange(self.cslice[0].start,self.cslice[0].stop))
            cond = clusters[self.cslice]==self.label
            
            #Mask condition
            if mask is not None:
                cond *= ~mask[self.cslice] 
            
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
    
    
    
    def fit_1Dsersic(self, data, segmap, uiso, ps, mzero, dr=5, Rmax=250, mask=None,
                     minpts=5, makeplots=False, p0=None, pix_corr=2, Nreps=1,
                     **kwargs):
        
        from scipy.optimize import curve_fit
        from . import sersic
        
        default_return = {'x0':None,
                        'y0':None,
                        'theta':None,
                        'q':None, 
                        'ue':None,
                        're':None,
                        'n':None,
                        'mag':None,
                        'due':None,
                        'dre':None, 
                        'dn':None,
                        'dmag':None}
        
        #Get weighted ellipse
        e_weight = self.get_ellipse_rms_weighted(data, segmap, **kwargs)
        e_weight.x0 = int(e_weight.x0)
        e_weight.y0 = int(e_weight.y0)
        
        #Get data cutout
        xmin = int( np.max((0, int(e_weight.x0-Rmax))) )
        ymin = int( np.max((0, int(e_weight.y0-Rmax))) )
        xmax = int( np.min((int(e_weight.x0+Rmax), data.shape[1])) )
        ymax = int( np.min((int(e_weight.y0+Rmax), data.shape[0])) )
        
        cutout = data[ymin:ymax, xmin:xmax]
                    
        #Get mask cutout
        if mask is not None:
            mask_crp = mask[ymin:ymax, xmin:xmax]
        else:
            mask_crp = False
        
        #Define coordinate grid
        xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
                
        r = 0
        Icrit = SB.SB2Counts(uiso, ps, mzero)
        Is = []
        dIs = []
        rs = []
        while True:
            
            #Get inside ellipse condition
            if r == 0:
                e2 = geometry.ellipse(x0=e_weight.x0, y0=e_weight.y0, a=r+dr, b=(r+dr)*e_weight.q, theta=e_weight.theta)
                inside = e2.check_inside(xx,yy) * ~mask_crp
                    
            else:
                e1 = geometry.ellipse(x0=e_weight.x0, y0=e_weight.y0, a=r, b=r*e_weight.q, theta=e_weight.theta)
                e2 = geometry.ellipse(x0=e_weight.x0, y0=e_weight.y0, a=r+dr, b=(r+dr)*e_weight.q, theta=e_weight.theta)
            
                inside = e2.check_inside(xx,yy) * ~e1.check_inside(xx,yy) * ~mask_crp
                
            #Calculate average brightness within anulus
            I = np.median(cutout[inside])
                       
            #Calculate error taking into account pixel correlation
            dI = np.std(cutout[inside]) / np.sqrt(inside.sum() / np.pi) * pix_corr
            
            if np.isfinite(I) * (inside.sum() > 2):
                Is.append(I)
                dIs.append(dI)
                rs.append(0.5*(2*r + dr))
                
                #Break condition
                if (I <= Icrit) * (len(Is) >= minpts):
                    break
            #Increase the radius    
            r += dr
    
            #Max radius condition
            if r >= Rmax:
                print('WARNING: maximum radius has been reached.')
                if len(Is) < minpts:
                    return default_return
                break
                      
        #Do some data pertubations within error to get more robust result
        if Nreps > 1:
            
            from astropy.stats import sigma_clip
            if makeplots:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.errorbar(rs, Is, yerr=dIs, color='k')
            
            popts = []
            for n in range(Nreps):
                Is2 = np.random.normal(loc=Is, scale=dIs)
                try:
                    popt, pcov = curve_fit(sersic.profile, xdata=rs, ydata=Is2,
                                           p0=p0, bounds=(np.zeros(3),
                                                          np.ones(3)*np.inf))
                    popts.append(popt)
                    if makeplots:
                        rs2 = np.linspace(0, np.max(rs))
                        plt.plot(rs2, sersic.profile(rs2, popt[0], popt[1], popt[2]), color='grey')
                except:
                    pass
            
            if len(popts) == 0:    
                return default_return
        
            else:
                #Get median clipped result
                popts = np.vstack([popts])

                #Apply the sigma clip
                sigcs= [sigma_clip(popts[:,i]) for i in range(popts.shape[1])]
                popt = [np.median(sigcs[i]) for i in range(popts.shape[1])]
                perr = [np.std(sigcs[i]) for i in range(popts.shape[1])]
                
                if makeplots:
                    rs2 = np.linspace(0, np.max(rs))
                    plt.plot(rs2, sersic.profile(rs2, popt[0], popt[1], popt[2]), color='r')
                    
                #Return dictionary with result
                mag = sersic.magnitude(SB.Counts2SB(popt[0],ps,mzero), popt[1]*ps, popt[2])
                
                return {'x0':e_weight.x0,
                        'y0':e_weight.y0,
                        'q':e_weight.q,
                        'theta':e_weight.theta,
                        'ue':SB.Counts2SB(popt[0],ps,mzero),
                        're':popt[1]*ps,
                        'n':popt[2],
                        'mag':mag,
                        'due':SB.Counts2SB(popt[0]-perr[0],ps,mzero)-SB.Counts2SB(popt[0],ps,mzero),
                        'dre':perr[1]*ps,
                        'dn':perr[2],
                        'dmag':None
                        }
                        
        #If no repeats are necessary, do a single fit...
        else:
            if makeplots:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.errorbar(rs, Is, yerr=dIs, color='k')
            try:
                popt, pcov = curve_fit(sersic.profile, xdata=rs, ydata=Is, sigma=dIs,
                                       p0=p0)
                perr = np.sqrt(np.diag(pcov))
                
                if makeplots:
                    rs2 = np.linspace(0, np.max(rs))
                    plt.plot(rs2, sersic.profile(rs2, popt[0], popt[1], popt[2]), color='r')
                
                #Return dictionary with result
                mag = sersic.magnitude(SB.Counts2SB(popt[0],ps,mzero), popt[1]*ps, popt[2])
                
                return {'x0':e_weight.x0,
                        'y0':e_weight.y0,
                        'q':e_weight.q,
                        'theta':e_weight.theta,
                        'ue':SB.Counts2SB(popt[0],ps,mzero),
                        're':popt[1]*ps,
                        'n':popt[2],
                        'mag':mag,
                        'due':SB.Counts2SB(popt[0]-perr[0],ps,mzero)-SB.Counts2SB(popt[0],ps,mzero),
                        'dre':perr[1]*ps,
                        'dn':perr[2],
                        'dmag':None
                        }
            
            except Exception as e:
                print(e)
                return default_return
            
        

    def display(self, data, ax=None, mapping=np.arcsinh, mask=None, size=None,
                segmap=None, cmap='binary', apply_mask=True, **kwargs):
        '''
        Display the slice of data corresponding to the source.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        import matplotlib.pyplot as plt
        
        if mapping is None: mapping = lambda x: x
        
        if ax is None:
            fig, ax = plt.subplots()
        if size is None: 
            slc = self.cslice
        else:
            R = int(size/2)
            x0 = int(0.5 * (self.cslice[1].start + self.cslice[1].stop))
            y0 = int(0.5 * (self.cslice[0].start + self.cslice[0].stop))
            xovr = int(np.max((0, x0+R-data.shape[1])))
            xund = int(np.max((0, R-x0)))
            yovr = int(np.max((0, y0+R-data.shape[0])))
            yund = int(np.max((0, R-y0)))
            slc = ( slice(y0-R+yund, y0+R-yovr),
                    slice(x0-R+xund, x0+R-xovr) )
      
        if apply_mask:
            data2 = np.zeros((slc[0].stop-slc[0].start,slc[1].stop-slc[1].start ))
            data2[:,:] = data[slc]
            data2[mask[slc]] = float(np.nan)
            ax.imshow(mapping(data2), cmap=cmap, **kwargs)
            data3 = np.zeros((slc[0].stop-slc[0].start,slc[1].stop-slc[1].start ))
            data3[:,:] = data[slc]
            data3[~mask[slc]] = float(np.nan)
            ax.imshow(mapping(data3), cmap=cmap, **kwargs)
        else:
            ax.imshow(mapping(data[slc]), cmap=cmap, **kwargs)
            
        if mask is not None:
            if mask[slc].any():
                ax.contour(mask[slc], colors='deepskyblue')
        if segmap is not None:
            ax.contour(segmap[slc] == self.label, colors='lawngreen')
            try:
                ax.contour((segmap[slc]!=self.label) * (segmap[slc]!=0), colors='orange')
            except ValueError:
                pass
                
            
            
    def make_cutout(self, data, size=None, x0=None, y0=None, wcs=None, copy=True,
                    segmap=False, **kwargs):
        '''
        Make a cutout of the data.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        from astropy.nddata import Cutout2D
        
        if size is None: 
            size = int(np.max((self.cslice[0].stop-self.cslice[0].start,
                               self.cslice[1].stop-self.cslice[1].start)))
        if x0 is None:
            x0 = int(0.5 * (self.cslice[1].start + self.cslice[1].stop))
        else:
            x0 = int(x0)
        if y0 is None:
            y0 = int(0.5 * (self.cslice[0].start + self.cslice[0].stop))
        else:
            y0 = int(y0)
            
        if segmap:
            cutout = Cutout2D(data==self.label, (x0,y0), size, wcs=wcs, copy=copy,
                              **kwargs)
        else:
            cutout = Cutout2D(data, (x0,y0), size, wcs=wcs, copy=copy, **kwargs)
        
        return cutout
            
        