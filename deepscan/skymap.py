# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:57:08 2016

@author: Dan

Generation of noise map by finite differences
"""

import os, tempfile, time, gc
import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.restoration import inpaint
from multiprocessing import Pool
from . import sextractor, masking, NTHREADS, BUFFSIZE, dbscan


def _process_init(fp, interp):
    global smap
    global interp_
    smap = fp
    interp_ = interp
def gen_spline_map(points, values, sizex, sizey, fillwidth=None, Nthreads=NTHREADS,
                   lowmem=False):
    '''
    Fit a cubic spline to data.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    #Ensure finite
    cond_finite = np.isfinite(values)
    points = points[cond_finite]
    values = values[cond_finite]
    
    #If a small image
    if not lowmem:
        
        grid_x, grid_y = np.meshgrid(np.arange(0, sizex, 1), np.arange(0, sizey, 1))
        splined = griddata(points, values, (grid_x, grid_y), method='cubic').astype(values.dtype)
    
        
    else: #Low memory version
        
        #Decide how big the boxes are to fill in one go
        if fillwidth is None:
            fillwidth = int(BUFFSIZE/values.dtype.itemsize)
        
        #Calculate the interpolation
        interp = CloughTocher2DInterpolator(points, values)
        
        #Make a temporary directory
        pool = None
        with tempfile.TemporaryDirectory() as dirpath:
            try:
                #Create file name
                mapfile = os.path.join(dirpath, 'spline.dat')
                
                #Create memmap 
                splined = np.memmap(mapfile, dtype=values.dtype, mode='w+', shape=(sizey,sizex))
                #gcube[:] = np.zeros((sizey, sizex))
                
                #Calculate number of windows to fill
                nnx = int(np.ceil(sizex/fillwidth))
                nny = int(np.ceil(sizey/fillwidth))
                
                #Loop over the windows and calculate bounds for discrete part
                bounds = []
                for nx in range(nnx):
                    for ny in range(nny):
                        xmin = nx*fillwidth
                        xmax = np.min(( (nx+1)*fillwidth, sizex ))
                        ymin = ny*fillwidth
                        ymax = np.min(( (ny+1)*fillwidth, sizey ))
                        bounds.append([xmin, xmax, ymin, ymax])
                                    
                if Nthreads > 1:
                    #Create thread pool
                    pool = Pool(processes=Nthreads, initializer=_process_init, initargs=(splined,interp))
    
                    #Map the portions to the threads - returns None as modifying the array directly
                    pool.starmap(gen_spline_map_partial, bounds)
                    
                else:   #Non-threaded case
                    for b in bounds:
                        grid_x, grid_y = np.meshgrid(np.arange(b[0],b[1]),
                                                     np.arange(b[2],b[3]))
                        splined[b[2]:b[3],b[0]:b[1]] = interp(
                                np.vstack([grid_x.reshape(grid_x.size),
                            grid_y.reshape(grid_y.size)]).T).reshape(grid_x.shape)
                         
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
    return splined



def gen_spline_map_partial(xmin, xmax, ymin, ymax):
    '''
    Fill region of spline memmap.
    
    Paramters
    ---------
    
    Returns
    -------   
    
    '''    
    #Create a grid corresponding to the box
    grid_x, grid_y = np.meshgrid(np.arange(xmin,xmax), np.arange(ymin,ymax))
    
    #Calculate interpolated values
    interpd = interp_(np.vstack([grid_x.reshape(grid_x.size),
                                grid_y.reshape(grid_y.size)]).T).reshape(grid_x.shape)
    
    #Fill the memmap
    smap[ymin:ymax,xmin:xmax] = interpd
    


def rms_quantile(arr):
    '''
    Lower quantile estimate of the RMS.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    return np.median(arr) - np.percentile(arr, 15.9)



def get_spline_data(skygrid, rmsgrid, xmins, xmaxs, ymins, ymaxs):
    '''
    Organise data from 2D array for spline.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    gx, gy = np.meshgrid((xmins+xmaxs)/2, (ymins+ymaxs)/2)
    points = np.vstack([gx.reshape(gx.size), gy.reshape(gy.size)]).T
    values_n = rmsgrid.reshape(rmsgrid.size)
    values_b = skygrid.reshape(skygrid.size)
    return points, values_b, values_n

"""
def measure_sky(data, meshsize, mask=None, est_sky=np.median, est_rms=rms_quantile,
           fillfrac=0.5, lowmem=False, Nthreads=NTHREADS, medfiltsize=5,
           verbose=False):
    
    '''
    Estimate the sky and RMS in meshes and interpolate.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    #bgsamp = int(np.ceil(meshsize/6))
    bgsamp = int(np.ceil(meshsize/3))
    xmins = np.arange(0, data.shape[1], bgsamp)
    xmaxs = xmins + bgsamp
    xmins[-1] = data.shape[1] - bgsamp
    xmaxs[-1] = data.shape[1]
    ymins = np.arange(0, data.shape[0], bgsamp)
    ymaxs = ymins + bgsamp
    ymins[-1] = data.shape[0] - bgsamp
    ymaxs[-1] = data.shape[0]
    
    #Accounting for out of bounds interpolation
    xmins = np.hstack([[xmins[0]], xmins, [xmins[-1]]])
    xmaxs = np.hstack([[xmaxs[0]], xmaxs, [xmaxs[-1]]])
    ymins = np.hstack([[ymins[0]], ymins, [ymins[-1]]])
    ymaxs = np.hstack([[ymaxs[0]], ymaxs, [ymaxs[-1]]])

    t0 = time.time()
    
    #Fill the small grid ready for interpolating
    bggrid = np.zeros((ymaxs.size, xmaxs.size),dtype=data.dtype) * float(np.nan)
    rmsgrid = np.zeros((ymaxs.size, xmaxs.size),dtype=data.dtype) * float(np.nan)
    for i in range(xmaxs.size):
        for j in range(ymaxs.size):
            slc = (slice(ymins[j], ymaxs[j]), slice(xmins[i], xmaxs[i])) 
            
            #Crop the mask
            if mask is None:
                mask_crp=False
            else:
                mask_crp=mask[slc]
            
            #Ensure slice at least fillfrac full
            if np.sum(mask_crp) < fillfrac * (slc[0].stop-slc[0].start) * (slc[1].stop-slc[1].start):
                bggrid[j, i] = est_sky(data[slc][~mask_crp])
                rmsgrid[j,i] = est_rms(data[slc][~mask_crp])
                
    t1 = time.time() - t0
    if verbose: print('-measure_sky: small grid filled after %i secs.' % t1)
    
    
    #Inpaint the nan values
    nmax = np.max(rmsgrid[~np.isnan(rmsgrid)])   #Skimage needs numbers between -1 and 1 for float image
    nmin = np.min(rmsgrid[~np.isnan(rmsgrid)])
    rmsgrid = inpaint.inpaint_biharmonic((rmsgrid-nmin)/(nmax-nmin), np.isnan(rmsgrid))
    rmsgrid = rmsgrid*(nmax-nmin) + nmin

    bmax = np.max(bggrid[~np.isnan(bggrid)])   
    bmin = np.min(bggrid[~np.isnan(bggrid)])
    bggrid = inpaint.inpaint_biharmonic((bggrid-bmin)/(bmax-bmin), np.isnan(bggrid))
    bggrid = bggrid*(bmax-bmin) + bmin
    
    #Median filter the result
    if medfiltsize != 0:
        bggrid = median_filter(bggrid, medfiltsize)
        rmsgrid = median_filter(rmsgrid, medfiltsize)
    
    t2 = time.time() - t1 - t0
    if verbose: print('-measure_sky: small grid filtered after %i secs.' % t2)
    
    #Accounting for out of bounds interpolation
    xmins[0] = 0; xmins[-1]=data.shape[1]
    xmaxs[0] = 0; xmaxs[-1]=data.shape[1]
    ymins[0] = 0; ymins[-1]=data.shape[0]
    ymaxs[0] = 0; ymaxs[-1]=data.shape[0]
    
    #Get the data for spline
    points, values_b, values_n = get_spline_data(bggrid.astype(data.dtype),
                                                 rmsgrid.astype(data.dtype),
                                                 xmins, xmaxs,
                                                 ymins, ymaxs) 
    #Create the splines
    ndata = gen_spline_map(points, values_n, data.shape[1], data.shape[0],lowmem=lowmem,Nthreads=Nthreads)
    bdata = gen_spline_map(points, values_b, data.shape[1], data.shape[0],lowmem=lowmem,Nthreads=Nthreads)
    
    #Account for negative RMS
    ndata[ndata<0] = 0 
    
    t3 = time.time() - t2 - t1 - t0
    if verbose: print('-measure_sky: full grid filled after %i secs.' % t3)
    
    return bdata, ndata
"""

"""
def create_skymask(data, meshsize, sigma=None, its=5, sigma_clip=3, 
                   tol=1.05, verbose=False, lowmem=False, mask=False,
                   Nthreads=NTHREADS, medfiltsize=5, **kwargs):
    '''
    Create a source mask by iteritively rejecting high SNR peaks on smoothed
    image.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    if mask is None:
        mask = False
        
    #Get the initial crude estimate for masking 
    if sigma is not None:
        smooth = gaussian_filter(data, sigma)
    else:
        smooth = data
        
    bgvs, rmsvs = measure_sky(smooth, meshsize=meshsize, lowmem=lowmem,
                              Nthreads=Nthreads, verbose=verbose,
                              medfiltsize=medfiltsize, **kwargs)
    
    if lowmem:
        mask_s = np.zeros_like(data, dtype='bool')
        for row in range(data.shape[0]):
            mask_s[row]= (smooth[row]>bgvs[row]+sigma_clip*rmsvs[row]) 
            + (smooth[row]<bgvs[row]-sigma_clip*rmsvs[row])
    else:
        mask_s = (smooth>bgvs+sigma_clip*rmsvs) + (smooth<bgvs-sigma_clip*rmsvs)
    
    #Include the provided mask
    mask_s += mask
    
    masked_area = mask_s.sum()
    
    for i in range(int(its-1)):
    
        #Do a repeat with the mask in place
        bgvs, rmsvs = measure_sky(smooth, meshsize=meshsize, mask=mask_s, lowmem=lowmem,
                                  Nthreads=Nthreads, verbose=verbose,
                                  medfiltsize=medfiltsize, **kwargs)
        
        if lowmem:
            mask_s = np.zeros_like(data, dtype='bool')
            for row in range(data.shape[0]):
                mask_s[row]= (smooth[row]>bgvs[row]+sigma_clip*rmsvs[row]) 
                + (smooth[row]<bgvs[row]-sigma_clip*rmsvs[row])
        else:
            mask_s = (smooth>bgvs+sigma_clip*rmsvs) + (smooth<bgvs-sigma_clip*rmsvs)
        
        mask_s += mask
        
        #Masked area convergence
        masked_area_ = mask_s.sum()
        if masked_area_ / masked_area <= tol:
            if verbose:
                print('-source mask converged after %i iterations.' % (i+1))
            break
        masked_area = masked_area_
        
        if (i==its-2):
            print('-WARNING: skymap source mask did not converge.')
        
    return mask_s
"""


"""
def skymap(data, meshsize_initial, meshsize_final, mask=None, makeplots=False, 
           verbose=True, lowmem=False, clip=True, Nthreads=NTHREADS, 
           create_mask=False, sex_kwargs=None, ps=1, mzero=0, 
           **kwargs):
"""



"""
def skymap(data, meshsize, mask=None, makeplots=False, 
           verbose=True, lowmem=False, eps=5, kappa=5, Nthreads=NTHREADS, 
           create_mask=False, ps=1, mzero=0, **kwargs):    
    '''
    Create background and RMS maps.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    t0 = time.time()
    
    if (mask is None) * create_mask:
        
        sky, rms = measure_sky_iteritive(data, meshsize, mask=mask, lowmem=lowmem,
                           Nthreads=Nthreads, verbose=verbose, eps=eps, kappa=kappa,
                           ps=ps, **kwargs)
        
        #Create mask memmap
        tfile = tempfile.NamedTemporaryFile(delete=True)
        mask = np.memmap(tfile.name, shape=data.shape, dtype='bool', mode='w+')
        
        #Perform iteritive masking
        
        
        
        
    else:
        sky, rms = measure_sky(data, meshsize, mask=mask, lowmem=lowmem,
                           Nthreads=Nthreads, verbose=verbose, **kwargs)
        
    
    '''
    if create_mask:
        if verbose: print('skymap: creating source mask...')
        es_sex = sextractor.get_ellipses(data=data,mzero=mzero,ps=ps,mask=mask,
                                        verbose=verbose, Nthreads=Nthreads,
                                        debug=False, **sex_kwargs)
        if mask is None:
            mask = masking.mask_ellipses(np.zeros_like(data, dtype='bool'),
                                         es_sex, rms=None, fillval=1,
                                         Nthreads=Nthreads).astype('bool')
        else:
            mask = masking.mask_ellipses(mask, es_sex, rms=None, fillval=1,
                                         Nthreads=Nthreads).astype('bool')
                 
    if clip:
        if verbose: print('skymap: making sky clip...')
        mask_ = create_skymask(data, meshsize_initial, verbose=verbose, 
                          lowmem=lowmem, mask=mask, Nthreads=Nthreads,
                          **kwargs)
    
    else:
        if verbose: print('skymap: skipping sky clip...')
        mask_ = mask
    
    if verbose: print('skymap: measuring sky...')
    sky, rms = measure_sky(data, meshsize_final, mask=mask_, lowmem=lowmem,
                           Nthreads=Nthreads, verbose=verbose, **kwargs)
    
    '''
    if makeplots:
        import matplotlib.pyplot as plt
        masked = data.copy()
        masked[mask] = float(np.nan)
        plt.figure(); plt.imshow(masked); plt.title('masked')
        plt.figure(); plt.imshow(sky); plt.title('sky')
        plt.figure(); plt.imshow(rms); plt.title('RMS')
        
    t1 = time.time() - t0
    if verbose: print('skymap: finished after %i seconds.' % t1)

    return sky, rms   
"""    
    
    
    
    


class Mesh():
    def __init__(self, i, j, xmin, xmax, ymin, ymax,active=True,sky=np.nan,rms=np.nan):
        self.i = int(i)
        self.j = int(j)
        self.slc = (slice(ymin,ymax), slice(xmin,xmax))
        self.active=active
        self.sky = sky
        self.rms = rms
        self.area = (xmax-xmin) * (ymax-ymin)
    def measure(self, data, mask=None, fillfrac=0, estimator_sky=np.median,
                                        estimator_rms=rms_quantile):
        minpts = self.area * fillfrac
        minpts = np.max((minpts, 2))
        if mask is None:
            sky = estimator_sky(data[self.slc])
            rms = estimator_rms(data[self.slc])
        else:
            if np.sum(~mask[self.slc]) > 2:
                sky = estimator_sky(data[self.slc][~mask[self.slc]])
                rms = estimator_rms(data[self.slc][~mask[self.slc]])  
            else:
                sky = np.nan
                rms = np.nan
        return sky, rms
    
    def update(self, sky, rms):
        self.sky = sky
        self.rms = rms
            
    def deactivate(self):
        self.active = False
        
        
def interpolate(sky_, rms_, shape, xmins, xmaxs, ymins, ymaxs,
                lowmem=True, Nthreads=NTHREADS):
    
    #Get the data for spline
    points, values_b, values_n = get_spline_data(sky_, rms_, 
                                                 xmins, xmaxs,
                                                 ymins, ymaxs) 
    #Create the splines
    rms = gen_spline_map(points, values_n, shape[1], shape[0],lowmem=lowmem,
                         Nthreads=Nthreads)
    sky = gen_spline_map(points, values_b, shape[1], shape[0],lowmem=lowmem,
                         Nthreads=Nthreads)
    
    return sky, rms

        
        
def skymap(data, meshsize, ps=1, eps=5, kappa=5, thresh=0.5,
                          medfiltsize=3, mask=None,
                          estimator_sky=np.median, estimator_rms=rms_quantile,
                          lowmem=True, tol=1.03, Niters=6, Nthreads=NTHREADS,
                          return_mask=False, fillfrac=0.3, verbose=False):
    if verbose: print('skymap: measuring sky...')
    t0 = time.time()
    
    if mask is None: 
        mask = np.zeros_like(data, dtype='bool')
    else:
        mask = mask.astype('bool').copy()
       
    meshsize = int(np.ceil(meshsize))
    
    xmins = np.arange(0, data.shape[1], meshsize)
    xmaxs = xmins + meshsize
    xmins[-1] = data.shape[1] - meshsize
    xmaxs[-1] = data.shape[1]
    ymins = np.arange(0, data.shape[0], meshsize)
    ymaxs = ymins + meshsize
    ymins[-1] = data.shape[0] - meshsize
    ymaxs[-1] = data.shape[0]
    
    #Accounting for out of bounds interpolation (1)
    xmins = np.hstack([[xmins[0]], xmins, [xmins[-1]]])
    xmaxs = np.hstack([[xmaxs[0]], xmaxs, [xmaxs[-1]]])
    ymins = np.hstack([[ymins[0]], ymins, [ymins[-1]]])
    ymaxs = np.hstack([[ymaxs[0]], ymaxs, [ymaxs[-1]]])
    
    #Create small arrays
    sky_ = np.zeros((ymaxs.size, xmaxs.size),dtype=data.dtype) 
    rms_ = np.zeros((ymaxs.size, xmaxs.size),dtype=data.dtype) 
    
    #Create meshes
    meshes = []
    for i in range(len(xmins)):
        for j in range(len(ymins)):
            meshes.append(Mesh(i,j,xmins[i],xmaxs[i],ymins[j],ymaxs[j]))
            
    #Accounting for out of bounds interpolation (2)
    xmins[0] = 0; xmins[-1]=data.shape[1]
    xmaxs[0] = 0; xmaxs[-1]=data.shape[1]
    ymins[0] = 0; ymins[-1]=data.shape[0]
    ymaxs[0] = 0; ymaxs[-1]=data.shape[0]
     
    """
    #Fill small arrays
    for mesh in meshes:
        s_, r_ = mesh.measure(data, mask=mask, fillfrac=fillfrac,
           estimator_sky=estimator_sky, estimator_rms=estimator_rms)
        sky_[mesh.j,mesh.i], rms_[mesh.j,mesh.i] = s_, r_
        mesh.sky = s_
        mesh.rms = r_
    
    
    #Inpaint the nan values
    rms_finite = np.isfinite(rms_)
    if not rms_finite.all():    #Skimage needs numbers between -1 and 1 for 
        nmax = np.max(rms_[rms_finite]) 
        nmin = np.min(rms_[rms_finite])
        rms_ = inpaint.inpaint_biharmonic((rms_-nmin)/(nmax-nmin),~rms_finite)
        rms_ = rms_*(nmax-nmin) + nmin

    sky_finite = np.isfinite(sky_)
    if not sky_finite.all():    
        bmax = np.max(sky_[sky_finite])   
        bmin = np.min(sky_[sky_finite])
        sky_ = inpaint.inpaint_biharmonic((sky_-bmin)/(bmax-bmin),~sky_finite)
        sky_ = sky_*(bmax-bmin) + bmin
                
    #Median filter the result
    if medfiltsize > 1:
        sky_ = median_filter(sky_, medfiltsize)
        rms_ = median_filter(rms_, medfiltsize)
    
    #Perform the interpolation
    sky, rms = interpolate(sky_, rms_, data.shape, xmins, xmaxs, ymins, ymaxs,
                                   lowmem=lowmem, Nthreads=Nthreads)
        
    #Finish without creating the mask
    if Niters == 0:
        return sky, rms
    
    #Get the DBSCAN clusters
    mask = dbscan.dbscan(data, mask=None, sky=None, eps=eps, kappa=kappa,
                           thresh=thresh, ps=ps, rms=rms, verbose=False
                           ).segmap_dilate.astype('bool')
    """
    
    #Do the iterations
    it = 0
    finished = False
    while not finished:
        
        #it += 1
        finished = True
          
        for mesh in meshes:
            if mesh.active:
                
                finished = False
                
                s_, r_ = mesh.measure(data,mask=mask, fillfrac=fillfrac,
                                      estimator_sky=estimator_sky,
                                      estimator_rms=estimator_rms)
                                
                sky_[mesh.j,mesh.i], rms_[mesh.j,mesh.i] = s_, r_
                
                if s_/mesh.sky <= tol:
                    mesh.deactivate()
                    
                mesh.sky = s_
                mesh.rms = r_
                
        if not finished:

            #Inpaint the nan values
            rms_finite = np.isfinite(rms_)
            if not rms_finite.all():    
                nmax = np.max(rms_[rms_finite]) 
                nmin = np.min(rms_[rms_finite])
                rms_ = inpaint.inpaint_biharmonic((rms_-nmin)/(nmax-nmin),~rms_finite)
                rms_ = rms_*(nmax-nmin) + nmin
        
            sky_finite = np.isfinite(sky_)
            if not sky_finite.all():    
                bmax = np.max(sky_[sky_finite])   
                bmin = np.min(sky_[sky_finite])
                sky_ = inpaint.inpaint_biharmonic((sky_-bmin)/(bmax-bmin),~sky_finite)
                sky_ = sky_*(bmax-bmin) + bmin
        
            sky_ = median_filter(sky_, medfiltsize)
            rms_ = median_filter(rms_, medfiltsize)
                
            sky, rms = interpolate(sky_, rms_, data.shape, xmins,xmaxs,ymins,ymaxs,
                               lowmem=lowmem, Nthreads=Nthreads)
            rms[rms<0] = 0
            
            if Niters == 0: #Don't create a new mask
                t1 = time.time() - t0
                if verbose: print('skymap: skipping iterative masking.')
                if return_mask:
                    if verbose: print('skymap: finished after %i seconds.' % t1)
                    return sky, rms, mask
                if verbose: print('skymap: finished after %i seconds.' % t1)
                return sky, rms
            
            mask += dbscan.dbscan(data, mask=None, sky=sky, eps=eps, kappa=kappa,
                           thresh=thresh, ps=ps, rms=rms, verbose=False
                           ).segmap_dilate.astype('bool')
            gc.collect()
        
        if it >= Niters:
            print('WARNING: skymap reached max iterations.')
            finished = True
        it += 1
     
    t1 = time.time() - t0
    if return_mask:
        if verbose: print('skymap: finished after %i seconds.' % t1)
        return sky, rms, mask
    if verbose: print('skymap: finished after %i seconds.' % t1)
    return sky, rms
                
        
                    
                
                
                
            
            
        
    
                   
        
        
    
    
    
    
