# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:57:08 2016

@author: Dan

Generation of noise map by finite differences
"""

import os, tempfile, shutil, time
import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.restoration import inpaint
from multiprocessing import Pool
from deepscan import NTHREADS, BUFFSIZE


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
    
    #If a small image
    if not lowmem:
        
        grid_x, grid_y = np.meshgrid(np.arange(0, sizex, 1), np.arange(0, sizey, 1))
        splined = griddata(points, values, (grid_x, grid_y), method='cubic')
    
        
    else: #Low memory version
        
        #Decide how big the boxes are to fill in one go
        if fillwidth is None:
            fillwidth = int(BUFFSIZE/values.dtype.itemsize)
        
        #Calculate the interpolation
        interp = CloughTocher2DInterpolator(points, values)
        
        #Make a temporary directory
        dirpath = tempfile.mkdtemp()  
        pool = None
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
            
            try:
                shutil.rmtree(dirpath)
            except:
                print('shutil.rmtree error: Unable to remove %s.' % dirpath)
            
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


def measure_sky(data, meshsize, mask=None, est_sky=np.median, est_rms=rms_quantile,
           fillfrac=0.5, lowmem=False, Nthreads=NTHREADS, verbose=False):
    
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
    bggrid = np.zeros((ymaxs.size, xmaxs.size)) * float(np.nan)
    rmsgrid = np.zeros((ymaxs.size, xmaxs.size)) * float(np.nan)
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
    bggrid = median_filter(bggrid, 3)
    rmsgrid = median_filter(rmsgrid, 3)
    
    t2 = time.time() - t1 - t0
    if verbose: print('-measure_sky: small grid filtered after %i secs.' % t2)
    
    #Accounting for out of bounds interpolation
    xmins[0] = 0; xmins[-1]=data.shape[1]
    xmaxs[0] = 0; xmaxs[-1]=data.shape[1]
    ymins[0] = 0; ymins[-1]=data.shape[0]
    ymaxs[0] = 0; ymaxs[-1]=data.shape[0]
    
    #Get the data for spline
    points, values_b, values_n = get_spline_data(bggrid, rmsgrid, xmins, xmaxs,
                                                 ymins, ymaxs) 
    #Create the splines
    ndata = gen_spline_map(points, values_n, data.shape[1], data.shape[0],lowmem=lowmem,Nthreads=Nthreads)
    bdata = gen_spline_map(points, values_b, data.shape[1], data.shape[0],lowmem=lowmem,Nthreads=Nthreads)
    
    #Account for negative RMS
    ndata[ndata<0] = 0 
    
    t3 = time.time() - t2 - t1 - t0
    if verbose: print('-measure_sky: full grid filled after %i secs.' % t3)
    
    return bdata, ndata



def create_skymask(data, meshsize, sigma=5, its=5, sigma_clip=3, 
                   tol=1.05, verbose=False, lowmem=False, mask=False,
                   Nthreads=NTHREADS, **kwargs):
    '''
    Create a source mask by iteritively rejecting high SNR peaks on smoothed
    image.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    #Get the initial crude estimate for masking 
    smooth = gaussian_filter(data, sigma)
        
    bgvs, rmsvs = measure_sky(smooth, meshsize=meshsize, lowmem=lowmem,
                              Nthreads=Nthreads, **kwargs, verbose=verbose)
    
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
                                  Nthreads=Nthreads, verbose=verbose, **kwargs)
        
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



def skymap(data, meshsize_initial, meshsize_final, mask_init=False, makeplots=False, 
           verbose=True, lowmem=False, Nthreads=NTHREADS, **kwargs):
    
    '''
    Create background and RMS maps.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    t0 = time.time()
    
    if verbose: print('skymap: creating source mask...')
    mask = create_skymask(data, meshsize_initial, verbose=verbose, 
                          lowmem=lowmem, mask=mask_init, Nthreads=Nthreads,
                          **kwargs)
    
    if verbose: print('skymap: measuring sky...')
    sky, rms = measure_sky(data, meshsize_final, mask=mask, lowmem=lowmem,
                           Nthreads=Nthreads, verbose=verbose, **kwargs)
    
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
    
    
    
    
    
    
    
    
