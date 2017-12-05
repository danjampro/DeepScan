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
from functools import partial
from multiprocessing import Pool
from . import NTHREADS, BUFFSIZE


def _process_init(fp):
    global smap
    smap = fp
def gen_spline_map(points, values, sizex, sizey, fillwidth=None, Nthreads=NTHREADS,
                   partition=True):
    '''
    Fit a cubic spline to data.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    #If a small image
    if not partition:
        
        grid_x, grid_y = np.meshgrid(np.linspace(0, sizex, sizex), np.linspace(0, sizey, sizey))
        gcube = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    else:
        
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
            gcube = np.memmap(mapfile, dtype='float32', mode='w+', shape=(sizey,sizex))
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
                    
            #Create thread pool
            pool = Pool(processes=Nthreads, initializer=_process_init, initargs=(gcube,))
            
            #Create partial function for gen_spline_map_partial
            partial_func = partial(gen_spline_map_partial, interp=interp)
                    
            #Map the portions to the threads - returns None as modifying the array directly
            pool.starmap(partial_func, bounds)
            
            #Save the completed memmap 
            #result = np.array(gcube)
            result = gcube
            
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            #del gcube
            shutil.rmtree(dirpath)
            
    return result



def gen_spline_map_partial(xmin, xmax, ymin, ymax, interp):
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
    interpd = interp(np.vstack([grid_x.reshape(grid_x.size),
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
           fillfrac=0.5):
    
    '''
    Estimate the sky and RMS in meshes and interpolate.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    bgsamp = int(np.ceil(meshsize/6))
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
            
            #Ensure slice at least half full
            if np.sum(mask_crp) < fillfrac * (slc[0].stop-slc[0].start) * (slc[1].stop-slc[1].start):
                bggrid[j, i] = est_sky(data[slc][~mask_crp])
                rmsgrid[j,i] = est_rms(data[slc][~mask_crp])
    
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
    
    #Accounting for out of bounds interpolation
    xmins[0] = 0; xmins[-1]=data.shape[1]
    xmaxs[0] = 0; xmaxs[-1]=data.shape[1]
    ymins[0] = 0; ymins[-1]=data.shape[0]
    ymaxs[0] = 0; ymaxs[-1]=data.shape[0]
    
    points, values_b, values_n = get_spline_data(bggrid, rmsgrid, xmins, xmaxs,
                                                 ymins, ymaxs) 
    ndata = gen_spline_map(points, values_n, data.shape[1], data.shape[0])
    bdata = gen_spline_map(points, values_b, data.shape[1], data.shape[0])
    ndata[ndata<0] = 0 #Account for negative RMS
    
    return bdata, ndata



def create_skymask(data, meshsize, sigma=5, its=5, high=1.5, low=5, 
                   tol=1.05, verbose=False, **kwargs):
    
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
    bgvs, rmsvs = measure_sky(smooth, meshsize=meshsize, **kwargs)
    mask_s = (smooth>bgvs+high*rmsvs) + (smooth<=bgvs-low*rmsvs)
    
    masked_area = mask_s.sum()
    
    for i in range(int(its-1)):
    
        #Do a repeat with the mask in place
        bgvs, rmsvs = measure_sky(smooth, meshsize=meshsize, mask=mask_s, **kwargs)
        mask_s = (smooth>bgvs+high*rmsvs) + (smooth<=bgvs-low*rmsvs)
        
        #Masked area convergence
        masked_area_ = mask_s.sum()
        if masked_area_ / masked_area <= tol:
            if verbose:
                print('-source mask converged after %i iterations.' % (i+1))
            break
        masked_area = masked_area_
        
        if i==its-2:
            print('-WARNING: skymap source mask did not converge.')
        
    return mask_s



def skymap(data, meshsize_initial, meshsize_final, mask=None, makeplots=False, 
           verbose=True, **kwargs):
    
    '''
    Create background and RMS maps.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    t0 = time.time()
    
    if mask is None:
        if verbose:
            print('skymap: creating source mask...')
        mask = create_skymask(data, meshsize_initial, verbose=verbose, **kwargs)
    
    if verbose:
        print('skymap: measuring sky...')
    sky, rms = measure_sky(data, meshsize_final, mask=mask, **kwargs)
    
    if makeplots:
        import matplotlib.pyplot as plt
        masked = data.copy()
        masked[mask] = float(np.nan)
        plt.figure(); plt.imshow(masked); plt.title('masked')
        plt.figure(); plt.imshow(sky); plt.title('sky')
        plt.figure(); plt.imshow(rms); plt.title('RMS')
        
    t1 = time.time() - t0
    print('skymap: finished after %i seconds.' % t1)

    return sky, rms   
    
    
    
    
    
    
    
    
""" 

from astropy.convolution import convolve, Tophat2DKernel
from scipy.stats import sigmaclip 

def get_bounds_basic(sizex, sizey, wsize, i, j, offsetx=1, offsety=0):
    
    ''' Return [xlow, xhigh, ylow, yhigh] for window '''
    
    #Initial window bound calculations for primary window
    xmin1 = i * wsize
    xmax1 = np.min((sizex, (i+1) * wsize))
    ymin1 = j * wsize
    ymax1 = np.min((sizey, (j+1) * wsize))
        
    return  [int(xmin1), int(xmax1), int(ymin1), int(ymax1)]

   
def measure_bg_basic(data, mask, bounds, default_std=0., default_bg=0.):
    
    '''Calculate standard deviation of unmasked region within bounds'''
    
    crp_data = data[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    
    if mask is not None:
        
        crp_mask = mask[bounds[2]:bounds[3], bounds[0]:bounds[1]]
        
        if not mask.any():
            return default_bg, default_std
        else:
            return np.median(crp_data[~crp_mask]), np.std(crp_data[~crp_mask])
    else:
        return np.median(crp_data)

            
def get_bounds_finite(sizex, sizey, wsize, i, j, offsetx=1, offsety=0):
    
    ''' Return [xlow, xhigh, ylow, yhigh] for primary and x-offset windows '''
    
    #Initial window bound calculations for primary window
    xmin1 = i * wsize
    xmax1 = np.min((sizex, (i+1) * wsize))
    ymin1 = j * wsize
    ymax1 = np.min((sizey, (j+1) * wsize))
    
    #Offset in x coordinates for offset window
    xmin2 = xmin1+offsetx
    xmax2 = xmax1+offsetx
    ymin2 = ymin1+offsety
    ymax2 = ymax1+offsety
    
    overlap=0
    
    #Boundary checking/cropping
    if offsetx > 0:
        xmax2_ = np.min((sizex, xmax2))
        if xmax2_ != xmax2:
            overlap = xmax2 - xmax2_
            xmax2 = xmax2_
            xmax1 = xmax1 - overlap
    elif offsetx < 0:
        xmin2_ = np.max((0, xmin2))
        if xmin2 != xmin2_:
            overlap = xmin2_ - xmin2
            xmin2 = xmin2_
            xmin1 = xmin1 + overlap
    if offsety > 0:
        ymax2_ = np.min((sizey, ymax2))
        if ymax2_ != ymax2:
            overlap = ymax2 - ymax2_
            ymax2 = ymax2_
            ymax1 = ymax1 - overlap
    elif offsety < 0:
        ymin2_ = np.max((0, ymin2))
        if ymin2 != ymin2_:
            overlap = ymin2_ - ymin2
            ymin2 = ymin2_
            ymin1 = ymin1 + overlap
    
    #print(offsetx, offsety, overlap, xmin1, xmax1, xmin2, xmax2, '...', ymin1, ymax1, ymin2, ymax2)
        
    return  [int(xmin1), int(xmax1), int(ymin1), int(ymax1)], \
            [int(xmin2), int(xmax2), int(ymin2), int(ymax2)]
    


def measure_bg_finite(data, bounds1, bounds2, warea, sig=3, thinning_factor=1, mask=None, fillfrac=0.25):
    
    ''' Calculate the standard width of the pixel-to-pixel noise'''
    
    #Crop data
    crp1 = data[bounds1[2]:bounds1[3]:thinning_factor, bounds1[0]:bounds1[1]:thinning_factor]
    crp2 = data[bounds2[2]:bounds2[3]:thinning_factor, bounds2[0]:bounds2[1]:thinning_factor]
            
    #If there are some non zero values...
    if len(crp1) > 0:
        
        #Calculate difference
        if mask is None:
            mask = ((crp1 == 0) * (crp2 == 0)) 
            diff = (crp1 - crp2)[~mask]
        else:
            m1 = mask[bounds1[2]:bounds1[3]:thinning_factor, bounds1[0]:bounds1[1]:thinning_factor]
            m2 = mask[bounds2[2]:bounds2[3]:thinning_factor, bounds2[0]:bounds2[1]:thinning_factor]
            m1 += (crp1 == 0)
            m2 += (crp2 == 0)
            diff = (crp1 - crp2)[~(m1+m2)]
            
        #Calculate std of difference (pixel to pixel noise)
        clipped = sigmaclip(diff, high=sig, low=sig)[0]
        std = np.std(clipped)
        
        #Also calculate a background value
        bg = np.median(clipped)
        
        #Require a certain fraction of pixels to be unmasked
        if len(diff) < fillfrac*warea:
            std = 0
            bg = 0
        
        #Don't accept nan values
        if np.isnan(std)+np.isnan(bg):
            std = 0
            bg = 0
            
    #Else if pure border, set std to 0     
    else:
        std = 0
        bg = 0
        
    #Accound for sqrt(2) factor
    std *= 1./np.sqrt(2)
    
    return bg, std
    
    
def skymap(data, wsize, sig=3, offset=1, thinning_factor=1, mode='quad', mask=None, fillfrac=0.25):
    
    ''' Estimate the background in wsize*wsize windows as write to fits '''
    
    mask = mask.astype('bool')
      
    #Measure image
    sizey, sizex = data.shape
    warea = wsize**2
    
    #Get indices for windows 
    nx = int(np.ceil(sizex / wsize))
    ny = int(np.ceil(sizey / wsize))
    
    #This is a map of the noise with resolution elements of wsize
    nmap_small = np.zeros((ny, nx))
    bmap_small = np.zeros((ny, nx))
    
    #Loop over windows 
    for i in range(nx):
        for j in range(ny):
            
            #2-directional symmetical finite difference
            if mode == 'quad':
                
                #Use a 'star' pattern
                stds = []
                combs = [(1,0),(-1,0),(0,1),(0,-1)]
                               
                for comb in combs:
                    #Calculate window bounds
                    bounds1, bounds2 = get_bounds_finite(sizex, sizey, wsize, i, j, offsetx=comb[0], offsety=comb[1])
            
                    #Calculate standard deviation (noise)
                    bg, std = measure_bg_finite(data, bounds1, bounds2, warea=warea, sig=sig, thinning_factor=thinning_factor,
                                                mask=mask, fillfrac=fillfrac)
                    stds.append(std)
                    bmap_small[j, i] = bg
                        
                #Update map
                if np.sum(stds) != 0:
                    nmap_small[j, i] = np.median([s for s in stds if s!=0])
                else:
                    nmap_small[j, i] = 0
                
                
            #Simple mean and standard deviation case
            elif mode == 'std':
                
                #Calculate std in non-masked regions directly
                bounds = get_bounds_basic(sizex, sizey, wsize, i, j)
                
                bmap_small[j, i], nmap_small[j, i] = measure_bg_basic(data, mask, bounds)
                
                
    #Store as 32 bit numbers to save memory
    nmap_small = nmap_small.astype(np.float32)
    bmap_small = bmap_small.astype(np.float32)
    
    #Make grid for interpolation
    gx, gy = np.meshgrid(np.linspace(0, sizex, nx), np.linspace(0, sizey, ny))
    
    #Mask 0s to eliminate boarder biasing
    mask = nmap_small != 0
        
    #Account for holes in data by mixture of tophat filter, inpaint, and median filter 
    kernel = Tophat2DKernel(1)
    nmap_small[nmap_small==0] = float(np.nan)
    nmap_small = convolve(nmap_small, kernel=kernel, boundary='extend')
    nmap_small[np.isnan(nmap_small)] = 0
        
    #Inpainting
    nmax = np.max(nmap_small)   #Skimage needs numbers between -1 and 1 for float image
    nmin = np.min(nmap_small)
    nmap_small = inpaint.inpaint_biharmonic((nmap_small-nmin)/(nmax-nmin), ~mask)
    nmap_small = nmap_small*(nmax-nmin) + nmin
    
    bmax = np.max(bmap_small)   
    bmin = np.min(bmap_small)
    bmap_small = inpaint.inpaint_biharmonic((bmap_small-bmin)/(bmax-bmin), ~mask)
    bmap_small = bmap_small*(bmax-bmin) + bmin
        
    #Median fitering
    nmap_small = median_filter(nmap_small, 3)
    bmap_small = median_filter(bmap_small, 3)
        
    #Get points and values for interpolation over full-size image
    points = np.vstack([gx.reshape(gx.size), gy.reshape(gy.size)]).T
    values_n = nmap_small.reshape(nmap_small.size)
    values_b = bmap_small.reshape(bmap_small.size)
        
    #Generate spline data on full-size image
    ndata = gen_spline_map(points, values_n, sizex, sizey)
    ndata = ndata.astype(np.float32)
    bdata = gen_spline_map(points, values_b, sizex, sizey)
    bdata = bdata.astype(np.float32)
    
    return bdata, ndata

#==============================================================================

def main(fitsdata, ofile_bg, ofile_rms, wsize, sig=3, offset=1, thinning_factor=1, overwrite=True,
          mode='quad', fitsmask=None, fillfrac=0.25, extension_mask=0, extension_data=0):
    
    '''Read the data, measure the nosie and save as FITS'''
    
    from . import utils
    
    #Read fits data
    data = utils.read_fits(fitsdata, extension=extension_data)
    
    #Read mask if necessary
    if fitsmask is not None:
        mask = utils.read_fits(fitsmask, extension=extension_mask)
    else:
        mask = None
    
    #Measure the noise
    bdata, ndata = skymap(data, wsize, sig=sig, thinning_factor=thinning_factor,
                        mask=mask, fillfrac=fillfrac, mode=mode, offset=offset )

    #Save the result
    utils.save_to_fits(bdata, ofile_bg,  overwrite=overwrite)
    utils.save_to_fits(ndata, ofile_rms, overwrite=overwrite)
    
"""