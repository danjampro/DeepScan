#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:48:49 2017

@author: danjampro
"""

import numpy as np
from scipy.ndimage.measurements import label, find_objects
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.nddata import Cutout2D
from . import geometry, SB, masking, utils, sersic


class Source():
    
    def __init__(self, label, slc):
        self.slc = slc
        self.label = label        
        self.subsources = []
        self.xcen = 0.5 * (slc[1].stop + slc[1].start)
        self.ycen = 0.5 * (slc[0].stop + slc[0].start)
    
    
    def get_crds(self, segmap, mask=None, store=False):
        '''
        Get the coordinates within the segmap.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''    
        xs, ys = np.meshgrid(np.arange(self.slc[1].start,self.slc[1].stop),
                             np.arange(self.slc[0].start,self.slc[0].stop))
        cond = segmap[self.slc]==self.label
            
        #Mask condition
        if mask is not None:
            cond *= ~mask[self.slc] 
            
        xs = xs[cond]
        ys = ys[cond]
                
        return xs, ys
    
    
    def crd_global2local(self, x, y):
        '''
        Convert coordinates on the full image to those in the slice.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        return x-self.slc[1].start, y-self.slc[0].start
    
    
    def crd_local2global(self, x, y):
        '''
        Convert coordinates on the full image to those in the slice.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        return x+self.slc[1].start, y+self.slc[0].start
    
    
    def get_data(self, data, segmap, sky=None, mask=None):
        '''
        Get the data corresponding to the segmap.
        
        Paramters
        ---------
        
        Returns
        -------
        
        '''
        if mask is not None:
            mask = mask.astype('bool')
            Is = data[self.slc][(segmap[self.slc]==self.label)*(
                                                        ~mask[self.slc])]  
        else:           
            Is = data[self.slc][segmap[self.slc]==self.label]
        
        if sky is not None:
            if hasattr(sky, '__len__'):
                Is -= sky[self.slc][(segmap[self.slc]==self.label)*(
                                                        ~mask[self.slc])]
            else:
                Is -= sky
        return Is
    
    
    def get_ellipse_max(self, segmap, mask=None):
        '''
        Fit ellipse with unweighted first and second order moments and rescale
        to the full size of the cluster.
        
        Paramters
        ---------
        
        Returns
        -------
        
        '''
        xs, ys = self.get_crds(segmap, mask=mask)
        return geometry.fit_ellipse(xs,ys,weights=None,rms=False)
    
    
    def get_ellipse_rms(self, segmap, mask=None):
        '''
        Fit ellipse with unweighted first and second order moments.
        
        Paramters
        ---------
        
        Returns
        -------
        
        '''
        xs, ys = self.get_crds(segmap, mask=mask)        
        return geometry.fit_ellipse(self.xs,self.ys,weights=None,rms=True)
    
    
    def get_ellipse_rms_weighted(self, data, segmap, mask=None, x0=None,
                                 y0=None, sky=None, weight_transform=None):
        '''
        Fit ellipse with weighted first and second order moments.
        
        Paramters
        ---------
        
        Returns
        -------
        
        '''        
        if weight_transform is None:
            weight_transform = lambda x: x
                
        Is = self.get_data(data, segmap, mask=mask, sky=sky)
        
        xs, ys = self.get_crds(segmap, mask=mask)
        
        xs = xs[Is>0] #Do not allow negative flux for weighting
        ys = ys[Is>0]
        Is = Is[Is>0]
        Is = weight_transform(Is)
        
        return geometry.fit_ellipse(xs,ys,weights=Is,rms=True,x0=x0,y0=y0)
    
    
    def flux(self, data, segmap, sky=None, mask=None):
        '''
        Sum the flux in the segmap.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        return np.sum(self.get_data(data,segmap=segmap,mask=mask,sky=sky))  

    
    def masked_flux(self, data, segmap, mask, sky=None):
        '''
        Sum the flux in the masked regions of the segmap.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        d1 = self.flux(data,segmap=segmap,mask=mask,sky=sky)
        d2 = self.flux(data,segmap=segmap,mask=None,sky=sky)
        return d2 - d1         

    
    def area(self, segmap, mask=None):
        '''
        Measure number of pixels in the masked segmap.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        return self.get_data(data=segmap, segmap=segmap, mask=mask).size
    

    def masked_area(self, segmap, mask):
        '''
        Measure number of masked pixels within the segmap.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        return self.area(segmap,mask=None) - self.area(segmap,mask=mask)
    
    
    def colour(self, data1, data2, segmap, mask=None, sky1=None,
                       sky2=None):
        '''
        Measure the colour (mag1 - mag2).
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        d1 = self.flux(data1, segmap=segmap, mask=mask, sky=sky1)
        d2 = self.flux(data2, segmap=segmap, mask=mask, sky=sky2)
        
        return -2.5*np.log10(d1/d2)
    
    
    """
    def asymmetry(self, data, segmap, mask=None, sky=None, axis='major'):
        '''
        Measure the asymmetry in the flux.
        '''
        xx, yy = np.meshgrid(np.arange(self.slc[1].start,self.slc[1].stop),
                             np.arange(self.slc[0].start,self.slc[0].stop))
        e = self.get_ellipse_max(segmap)
        v = e.principle_components()
    """
    
    
    def get_azdata(self, data, segmap, x0, y0, q, theta, mask=None, sky=None,
                   dr=1, tol=1.01, Rmax=1000, Rmin=5, verbose=True):
        
        #Get data cutout
        xmin = int( np.max((0, int(x0-Rmax))) )
        ymin = int( np.max((0, int(y0-Rmax))) )
        xmax = int( np.min((int(x0+Rmax), data.shape[1])) )
        ymax = int( np.min((int(y0+Rmax), data.shape[0])) )
        
        slc = (slice(ymin,ymax), slice(xmin,xmax))
        cutout = np.zeros((ymax-ymin,xmax-xmin))
        
        cutout[:,:] = data[slc]
        
        if sky is not None:
            cutout -= sky[slc]
                    
        #Get mask cutout
        if mask is not None:
            mask_crp = mask[slc]
        else:
            mask_crp = np.zeros_like(cutout,dtype='bool')
        mask_crp += np.isin(segmap[slc], [0, self.label], invert=True)
                
        xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))         
        r = 0
        Is = []; dIs = []; rs = []
        flux_tot = None
        while True:
            
            #Get inside ellipse condition
            if r == 0:
                e2 = geometry.Ellipse(x0=x0, y0=y0, a=r+dr, b=(r+dr)*q, theta=theta)
                inside_ = e2.check_inside(xx,yy) 
                inside = inside_ * ~mask_crp
                    
            else:
                e1 = geometry.Ellipse(x0=x0, y0=y0, a=r, b=r*q, theta=theta)
                e2 = geometry.Ellipse(x0=x0, y0=y0, a=r+dr, b=(r+dr)*q, theta=theta)
            
                inside_ = e2.check_inside(xx,yy) * ~e1.check_inside(xx,yy)
                inside = inside_ * ~mask_crp
                
            #Calculate average brightness within anulus
            I = np.median(cutout[inside])
            flux = I * inside_.sum()
            
            #Calculate standard error
            dI = np.std(cutout[inside]) / np.sqrt(inside.sum()) 
            
            if np.isfinite(I) * (inside.sum() > 2):
                Is.append(I)
                dIs.append(dI)
                rs.append(0.5*(2*r + dr))
                
                if flux_tot is None:
                    flux_tot = flux
                    
                #Break condition
                if (((flux_tot + flux)/flux_tot)<=tol) * (rs[-1] >= Rmin):
                    break
                else:
                    flux_tot += flux
                    
            #Increase the radius    
            r += dr
    
            #Max radius condition
            if r >= Rmax:
                if verbose:
                    print('WARNING: maximum radius has been reached.')
                break
            
        return np.array(rs), np.array(Is), np.array(dIs)

        

                           
    def fit_1Dsersic(self, data, segmap, ps, mzero, dr=1, Rmax=250, mask=None,
                     minpts=5, makeplots=False, ue0=None, re0=None, n0=None,
                     tol=1.05, smooth_size=None, verbose=False, mask_radius=None,
                     skybox=None, sky=None, pix_corr=1, Nreps=1, **kwargs):
                
        default_return = {'x0':self.xcen,
                        'y0':self.ycen,
                        'q':np.nan,
                        'theta':np.nan,
                        'ue':np.nan,
                        're':np.nan,
                        'n':np.nan,
                        'mag':np.nan,
                        'due':np.nan,
                        'dre':np.nan,
                        'dn':np.nan,
                        'dmag':np.nan
                        }
        
        #First estimate the centroid postition 
        if smooth_size is None:
            x0 = None; y0 = None
        else:
            #Esitmate centroid on smoothed frame
            dat = np.zeros((self.slc[0].stop-self.slc[0].start,
                            self.slc[1].stop-self.slc[1].start))
            dat[:,:] = data[self.slc][:,:]
            if mask is not None:
                dat[mask[self.slc]] = float(np.nan)
            dat[segmap[self.slc]!=self.label] = 0
            dat = convolve_fft(dat, Gaussian2DKernel(smooth_size))
            y0, x0 = np.unravel_index(np.argmax(dat, axis=None), dat.shape)
            x0 += self.slc[1].start
            y0 += self.slc[0].start
            
        #Estimate elliptical paramters
        e_weight = self.get_ellipse_rms_weighted(data, segmap, mask=mask, x0=x0, y0=y0, **kwargs)
        e_weight.x0 = int(e_weight.x0)
        e_weight.y0 = int(e_weight.y0)
        
        #Get data cutout
        xmin = int( np.max((0, int(e_weight.x0-Rmax))) )
        ymin = int( np.max((0, int(e_weight.y0-Rmax))) )
        xmax = int( np.min((int(e_weight.x0+Rmax), data.shape[1])) )
        ymax = int( np.min((int(e_weight.y0+Rmax), data.shape[0])) )
        cutout = np.zeros((ymax-ymin,xmax-xmin))
        cutout[:,:] = data[ymin:ymax, xmin:xmax]
        
        if sky is not None:
            cutout -= sky[ymin:ymax, xmin:xmax]
        
        #Estimate the local sky level
        if skybox is not None:
            skybox = int(np.ceil(skybox))
            xmin_ = int( np.max((0, int(e_weight.x0-skybox))) )
            ymin_ = int( np.max((0, int(e_weight.y0-skybox))) )
            xmax_ = int( np.min((int(e_weight.x0+skybox), data.shape[1])) )
            ymax_ = int( np.min((int(e_weight.y0+skybox), data.shape[0])) )
            slc = (slice(ymin_, ymax_), slice(xmin_, xmax_))
            try:
                if mask is not None:
                    mask_ = mask[slc] * ~np.isfinite(data[slc])
                else:
                    mask_ = ~np.isfinite(data[slc])
                sky_local = np.median(data[slc][(segmap[slc]==0)*(~mask_)])
            except:
                print('WARNING: Local sky estimate could not be made.')
                sky_local = 0
            if not np.isfinite(sky_local):
                print('WARNING: Local sky estimate could not be made.')
                sky_local = 0
            cutout -= sky_local
            
        #Get mask cutout
        if mask is not None:
            mask_crp = mask[ymin:ymax, xmin:xmax]
        else:
            mask_crp = np.zeros_like(cutout,dtype='bool')
        if mask_radius is not None:
            e_mask = geometry.Ellipse(a=mask_radius,b=mask_radius,
                                      x0=e_weight.x0-xmin,
                                      y0=e_weight.y0-ymin,
                                      theta=0)
            mask_crp = masking.mask_ellipses(mask_crp, [e_mask],
                                             rms=None,
                                             fillval=True).astype('bool')
        
        #Define coordinate grid
        xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
                
        r = 0
        Is = []
        dIs = []
        rs = []
        flux_tot = None
        while True:
            
            #Get inside ellipse condition
            if r == 0:
                e2 = geometry.Ellipse(x0=e_weight.x0, y0=e_weight.y0, a=r+dr, b=(r+dr)*e_weight.q, theta=e_weight.theta)
                inside_ = e2.check_inside(xx,yy) 
                inside = inside_ * ~mask_crp
                    
            else:
                e1 = geometry.Ellipse(x0=e_weight.x0, y0=e_weight.y0, a=r, b=r*e_weight.q, theta=e_weight.theta)
                e2 = geometry.Ellipse(x0=e_weight.x0, y0=e_weight.y0, a=r+dr, b=(r+dr)*e_weight.q, theta=e_weight.theta)
            
                inside_ = e2.check_inside(xx,yy) * ~e1.check_inside(xx,yy)
                inside = inside_ * ~mask_crp
                
            #Calculate average brightness within anulus
            I = np.median(cutout[inside])
            flux = I * inside_.sum()
            
            #Calculate error taking into account pixel correlation
            dI = np.std(cutout[inside]) / np.sqrt(inside.sum() / np.pi) * pix_corr
            
            if np.isfinite(I) * (inside.sum() > 2):
                Is.append(I)
                dIs.append(dI)
                rs.append(0.5*(2*r + dr))
                
                if flux_tot is None:
                    flux_tot = flux
                    
                #Break condition
                if (((flux_tot + flux)/flux_tot)<=tol) * (len(Is) >= minpts):
                    break
                else:
                    flux_tot += flux
                    
            #Increase the radius    
            r += dr
    
            #Max radius condition
            if r >= Rmax:
                if verbose:
                    print('WARNING: maximum radius has been reached.')
                if len(Is) < minpts:
                    return default_return
                break
        
        #Create initial guess parameter list
        if re0 is None:
            re0 = np.sqrt(np.sum(segmap[self.slc]==self.label) / np.pi) * ps
        if n0 is None:
            n0 = 1
        if ue0 is None:
            ue0 = sersic.meanSB2effectiveSB(SB.Counts2SB(self.flux(data=data,
                  segmap=segmap,mask=mask,sky=sky)/self.area(segmap=segmap,
                                                 mask=mask),ps,mzero),  re0,
                                                    n0, e_weight.q)
            
        #Rescale the intensity values - fit can fail otherwise
        Is = np.array(Is)
        dIs = np.array(dIs)
        A = 1./Is.max()
        Is *= A
        dIs *= A
        mzero += 2.5*np.log10(A)
        ue0 += 2.5*np.log10(A)
                
        #try:
        popt, perr = sersic.fit1D(rs=np.array(rs)*ps, Is=Is, dIs=dIs, re0=re0, ue0=ue0,
                                  n0=n0, ps=ps, mzero=mzero, **kwargs)
        
        if makeplots:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.errorbar(rs, Is/A, yerr=dIs/A, color='k')
            rs2 = np.linspace(0, np.max(rs))
            plt.plot(rs2, sersic.profile(rs2,SB.SB2Counts(popt[0],ps,mzero),
                                         popt[1]/ps, popt[2])/A, color='r')
            plt.ylim(-0.2/A,1.2/A)
        
        #Return dictionary with result            
        return {'x0':e_weight.x0,
                'y0':e_weight.y0,
                'q':e_weight.q,
                'theta':e_weight.theta,
                'ue':popt[0],
                're':popt[1],
                'n':popt[2],
                'mag':sersic.effectiveSB2mag(popt[0],popt[1],popt[2],e_weight.q),
                'due':perr[0],
                'dre':perr[1],
                'dn':perr[2],
                'dmag':None
                }

        

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
            slc = self.slc
        else:
            R = int(size/2)
            x0 = int(0.5 * (self.slc[1].start + self.slc[1].stop))
            y0 = int(0.5 * (self.slc[0].start + self.slc[0].stop))
            xovr = int(np.max((0, x0+R-data.shape[1])))
            xund = int(np.max((0, R-x0)))
            yovr = int(np.max((0, y0+R-data.shape[0])))
            yund = int(np.max((0, R-y0)))
            slc = ( slice(y0-R+yund, y0+R-yovr),
                    slice(x0-R+xund, x0+R-xovr) )
      
        #Mask         
        if (mask is not None):
            if mask[slc].any():
                if apply_mask:
                    data2 = np.zeros((slc[0].stop-slc[0].start,slc[1].stop-slc[1].start ))
                    data2[:,:] = data[slc]
                    data2[mask[slc]] = float(np.nan)
                    ax.imshow(mapping(data2), cmap=cmap, **kwargs)
                    data3 = np.zeros((slc[0].stop-slc[0].start,slc[1].stop-slc[1].start ))
                    data3[:,:] = data[slc]
                    data3[~mask[slc]] = float(np.nan)
                    ax.imshow(mapping(data3), cmap=cmap, **kwargs)
                    ax.contour(mask[slc], colors='deepskyblue',linewidths=0.2)
                else:
                    ax.imshow(mapping(data[slc]), cmap=cmap, **kwargs)
                    ax.contour(mask[slc], colors='deepskyblue',linewidths=0.2)
            else:
                ax.imshow(mapping(data[slc]), cmap=cmap, **kwargs)
         
        #Data
        else:
            ax.imshow(mapping(data[slc]), cmap=cmap, **kwargs)
        
        #Segmap
        if segmap is not None:
            ax.contour(segmap[slc] == self.label, colors='lawngreen',
                       linewidths=0.2)
            try:
                ax.contour((segmap[slc]!=self.label) * (segmap[slc]!=0),
                                               colors='orange',linewidths=0.2)
            except ValueError:
                pass
            
        #Subsources
        if len(self.subsources)!=0:
            smap = np.zeros((int(2*R-(yovr+yund)), int(2*R-(xovr+xund))),
                                                                dtype='bool')
            for ssrc in self.subsources:
                smap[ssrc.cslice[0].start - slc[0].start:
                     ssrc.cslice[0].stop - slc[0].start,
                     ssrc.cslice[1].start - slc[1].start:
                     ssrc.cslice[1].stop - slc[1].start] += ssrc.binary_mask
            ax.contour(smap, colors='pink',linewidths=0.2)
                
        #Central coordinate
        e_weight = self.get_ellipse_rms_weighted(data, segmap, mask=mask)
        e_weight.x0 = int(e_weight.x0-slc[1].start)
        e_weight.y0 = int(e_weight.y0-slc[0].start)
        plt.plot(e_weight.x0,e_weight.y0,color='m', marker='+')
        e_weight.draw(color='m', linewidth=0.2)
        
        
            
    def make_cutout(self, data, size=None, x0=None, y0=None, wcs=None, copy=True,
                    segmap=False, **kwargs):
        '''
        Make a cutout of the data.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        if size is None: 
            size = int(np.max((self.slc[0].stop-self.slc[0].start,
                               self.slc[1].stop-self.slc[1].start)))
        if x0 is None:
            x0 = int(0.5 * (self.slc[1].start + self.slc[1].stop))
        else:
            x0 = int(x0)
        if y0 is None:
            y0 = int(0.5 * (self.slc[0].start + self.slc[0].stop))
        else:
            y0 = int(y0)
            
        if segmap:
            cutout = Cutout2D(data==self.label, (x0,y0), size, wcs=wcs, copy=copy,
                              **kwargs)
        else:
            cutout = Cutout2D(data, (x0,y0), size, wcs=wcs, copy=copy, **kwargs)
        
        return cutout
    
    
    def get_subsources(self, data, segmap, mask, flux_ratio=0):
        '''
        Retrive masked components within segmap containing >flux_ratio times
        the flux of the original source with the mask in place.
        
        Parameters
        ----------
        data
        
        segmap
        
        mask
        
        flux_ratio
        
        Returns
        -------
        None
        
        '''
        #Make cutouts
        data_ = data[self.slc]
        segmap_ = segmap[self.slc]
        mask_ = mask[self.slc]
        
        #Label masked components and get slices
        masked = (mask_==1)*(segmap_==self.label)
        labeled, Nobj = label(masked.astype('int'))
        slices_all = find_objects(labeled)
        
        #Retrieve flux values for each masked object
        fluxes_mask = [np.sum(data_[labeled==i]) for i in range(1,Nobj+1)]
        
        #Retrieve flux of the original source with the mask applied
        flux0 = np.sum(data_[((segmap_==self.label)*(mask_==0))])
        
        #Identify significant components
        if flux0 > 0:
            scomps = [i for i in range(Nobj) if fluxes_mask[i]/flux0>=flux_ratio]
        else:
            scomps = [i for i in range(Nobj)]
            print('WARNING: Source has negative flux within segmap')
        
        #Create source objects and append to self.subsources
        slices_ = [slices_all[i] for i in scomps]
        
        #Rapply the initial coordinate offset to the selected slices
        slices = []
        for slc in slices_:
            slices.append((slice(slc[0].start+self.slc[0].start,
                                 slc[0].stop+self.slc[0].start), 
                           slice(slc[1].start+self.slc[1].start,
                                 slc[1].stop+self.slc[1].start)))
        #Create the subsources
        for i in range(len(scomps)):
            self.subsources.append( Subsource(slices[i],
                                              labeled[slices_[i]]==scomps[i]+1))
        
        
#==============================================================================

class Subsource(Source):
    def __init__(self, slc, binary_mask):
        Source.__init__(self, cslice=slc, label=1)
        self.binary_mask = binary_mask.astype('bool')
        
#==============================================================================
        
def get_sources(segmap):
    sources = []
    slices = find_objects(segmap)
    uids = np.arange(1, len(slices)+2)
    uids = [u for u, s in zip(uids,slices) if s is not None]
    slices = [s for s in slices if s is not None]
    for uid, s in zip(uids, slices):
        sources.append(Source(uid, s))
    return sources
            
        

#==============================================================================

class Cutout():
    
    def __init__(self, data, x0, y0, size, sky=None, rms=None, mask=None,
                 segmap=None, segmap_dilate=None, wcs=None, mode='partial',
                 seglabel=None, **kwargs):
        
        if wcs is None:
            self.data = Cutout2D(data,(x0,y0),size,copy=True,mode=mode,**kwargs).data
            self.header = None
        else:
            _ = Cutout2D(data,(x0,y0),size,copy=True,wcs=wcs,**kwargs)
            self.data = _.data
            self.header = _.wcs.to_header()
        
        if sky is None:
            self.sky = None
        else:
            self.sky = Cutout2D(sky,(x0,y0),size,wcs=wcs,copy=True,mode=mode,
                                                                **kwargs).data                             
        
        if rms is None:
            self.rms = None
        else:
            self.rms = Cutout2D(rms,(x0,y0),size,wcs=wcs,copy=True,mode=mode,
                                                                **kwargs).data    
        
        if mask is None:
            self.mask = None
        else:
            self.mask = Cutout2D(mask,(x0,y0),size,wcs=wcs,copy=True,mode=mode
                                                             ,**kwargs).data
        
        if segmap is None:
            self.segmap = None
        else:
            self.segmap = Cutout2D(segmap,(x0,y0),size,wcs=wcs,copy=True,
                                                   mode=mode, **kwargs).data
            if seglabel is not None:
                self.segmap=self.segmap==seglabel
            
            
        if segmap_dilate is None:
            self.segmap_dilate = None
        else:
            self.segmap_dilate = Cutout2D(segmap_dilate,(x0,y0),size,wcs=wcs,
                                          mode=mode, copy=True,**kwargs).data
            if seglabel is not None:
                self.segmap_dilate=self.segmap_dilate==seglabel
              
                                          
    def save(self, fname):
        '''
        Save the cutout as a multi-extension fits file.
        
        Paramters
        ---------
        
        Returns
        -------
        
        None.
        
        '''
        datas = [self.data]; headers = [self.header]
        
        if self.sky is not None:
            datas.append(self.sky)
            headers.append(self.header)
            headers[-1]['DATACAT'] = 'SKY'
        
        if self.rms is not None:
            datas.append(self.rms)
            headers.append(self.header)
            headers[-1]['DATACAT'] = 'RMS'
        
        if self.mask is not None:
            datas.append(self.mask.astype('float32'))
            headers.append(self.header)
            headers[-1]['DATACAT'] = 'MASK'
        
        if self.segmap is not None:
            datas.append(self.segmap.astype('float32'))
            headers.append(self.header)
            headers[-1]['DATACAT'] = 'SEG'
        
        if self.segmap_dilate is not None:
            datas.append(self.segmap_dilate.astype('float32'))
            headers.append(self.header)
            headers[-1]['DATACAT'] = 'SEGD'
        
        utils.save_to_MEF(datas=datas,headers=headers,fname=fname,
                          overwrite=True)
    

def load_cutout(fname):
    '''
    Load a data cutout.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    pass
        
       
#==============================================================================       
       