# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:57:08 2016

@author: Dan

Generation of noise map by finite differences
"""

import os, tempfile, time
import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from . import BUFFSIZE, dbscan

#==============================================================================

interpolators = {'cubic':CloughTocher2DInterpolator,
                 'linear':LinearNDInterpolator,
                 'nearest':NearestNDInterpolator}

#==============================================================================

def estimator_quantile(data):
    '''
    Median sky estimate + lower quantile RMS estimate.
    
    Paramters
    ---------
    data: 1D numpy array.
    
    Returns
    -------
    median (float): median sky estimate.
    
    quantile (float): lower quantile RMS estimate.    
    '''
    #Increasing order
    data = np.sort(data.reshape(data.size))
               
    #Get the CDF
    cdf = np.arange(data.size)/float(data.size)
            
    #Median and quantile estimate
    median = data[np.argmin(abs(cdf-0.5))]
    quantile = median - data[np.argmin(abs(cdf-0.159))]
    
    return median, quantile

#==============================================================================

class Mesh():
    
    def __init__(self, i, j, xmin, xmax, ymin, ymax, active=True, sky=None,
                 rms=None, fillfrac=0):
        
        #Indices on mesh grid
        self.i = int(i)
        self.j = int(j)
        
        #Corresponding slices on the image
        self.slc = (slice(ymin,ymax), slice(xmin,xmax))
        self.area = (xmax-xmin) * (ymax-ymin)
        
        #Initial values
        self.sky = sky if (sky is not None) else np.nan
        self.rms = rms if (rms is not None) else np.nan
        self.active=active
        
        #Minimum fullness required before interpolation
        self.fillfrac = fillfrac
        
    def measure(self, data, mask=None, skyfunc='median', rmsfunc='quantile'):
                
        #Require a minimum number of data points to proceed                        
        minpts = np.max((self.area*self.fillfrac, 2))
                    
        #Get the data
        data_ = data[self.slc]
        
        #Apply the mask if it exists
        if mask is not None:
            data_ = data_[mask[self.slc]<=0]
            
        #Check that minpts is satisfied
        if data_.size < minpts:
            return np.nan, np.nan          #Return nans

        #Apply estimators
        if (skyfunc=='median') or (rmsfunc=='quantile'):
            sky, rms = estimator_quantile(data_)
        if skyfunc != 'median':
            sky = skyfunc(data_)
        if rmsfunc != 'quantile':
            rms = rmsfunc(data_)
                    
        return sky, rms
        
    def update(self, sky, rms):      
        #Update the sky and rms values
        self.sky = sky
        self.rms = rms
            
    def deactivate(self):      
        #Deactivate the mesh
        self.active = False
        
#==============================================================================

def make_meshes(shape, meshsize, **kwargs):
    
    meshsize = int(np.ceil(meshsize))
    
    #Get mesh boundaries in pixel coordinates
    xmins = np.arange(0, shape[1], meshsize)
    xmaxs = xmins + meshsize
    ymins = np.arange(0, shape[0], meshsize)
    ymaxs = ymins + meshsize
    
    #Create the mesh grid
    meshes = []
    for x, (xmin, xmax) in enumerate(zip(xmins, xmaxs)):
        for y, (ymin, ymax) in enumerate(zip(ymins, ymaxs)):
            meshes.append(Mesh(i=x,j=y,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                               **kwargs)) 
    
    return meshes, xmins.size, ymins.size
            
#==============================================================================

def fill_nans(data):

    #Identify points requiring interpolation    
    isnan = ~np.isfinite(data)
    if not isnan.any():
        return data
    
    #Dilate to identify points used for interpolation
    isnan_dilate = binary_dilation(isnan) 
    
    #Get coordinates for interpolation
    yy, xx = np.where((isnan_dilate==1) & (isnan)==0)
    
    #Do the nearest value interpolation
    interp = NearestNDInterpolator(np.vstack([xx,yy]).T, data[yy,xx])
    
    #Fill in the holes
    yy, xx = np.where(isnan)
    data[yy, xx] = interp(np.vstack([xx,yy]).T)
    
    return data

#==============================================================================

def interpolate_meshgrid(data, shape, lowmem=False, method='cubic'):
        
    #Get coordinates for interpolation
    yy, xx = np.where(np.isfinite(data))    
    size0 = xx.size #Used later
    
    #Need to expand borders with same values
    xx2 = np.arange(0, data.shape[1]+1); xx2[-1] = data.shape[1]-1
    yy2 = np.arange(0, data.shape[0]+1); yy2[-1] = data.shape[0]-1
    
    #Bottom edge
    xx = np.hstack([xx, xx2])
    yy = np.hstack([yy, np.zeros_like(xx2)])
    #Left edge
    yy = np.hstack([yy, yy2])
    xx = np.hstack([xx, np.zeros_like(xx2)])
    #Right edge
    size1 = xx.size
    yy = np.hstack([yy, yy2])
    xx = np.hstack([xx, np.ones_like(xx2)*data.shape[1]-1])
    #Top edge
    size2 = xx.size
    xx = np.hstack([xx, xx2])
    yy = np.hstack([yy, np.ones_like(xx2)*data.shape[0]-1])
    
    #Get corresponding data
    zz = data[yy,xx]
    
    #Rectify the coordinates
    xx = xx.astype('float'); yy = yy.astype('float')
    xx[:size0] += 0.5; yy[:size0] += 0.5  
    xx[size1:size2] += 1
    yy[size2:] += 1  
    yy[-1] = data.shape[0]
    xx[-1] = data.shape[1]
        
    if not lowmem:
        
        #Get the grid to calculate interpolated values
        X, Y = np.meshgrid(np.linspace(0,data.shape[1],shape[1]),
                           np.linspace(0,data.shape[0],shape[0]))       
        #Do the interpolation
        interped = griddata(np.vstack([xx,yy]).T, zz, (X, Y), method=method)
        
    else:
        
        #Calculate the interpolation on memmap grid
        interped = interp_lowmem(np.vstack([xx,yy]).T, zz, data.shape, shape,
                                 method=method)            
    return interped
    
#==============================================================================

def interp_lowmem(points, values, shape_old, shape_new, fillwidth=None,
                  Nthreads=1, lowmem=False, method='cubic'):
    '''
    Fit a cubic spline to data.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
            
    #Decide how big the boxes are to fill in one go
    if fillwidth is None:
        fillwidth = int(BUFFSIZE/values.dtype.itemsize)
    
    #Calculate the interpolation
    points[:,0] *= shape_new[1]/shape_old[1]
    points[:,1] *= shape_new[0]/shape_old[0]
    interp = interpolators[method](points, values)
    
    #Make a memmap in a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
            
        #Create file name
        memmap = os.path.join(tempdir, 'spline.mmap')
        
        #Create memmap 
        interpd = np.memmap(memmap, dtype=values.dtype, mode='w+', 
                            shape=shape_new)
        
        #Calculate number of windows to fill
        nnx = int(np.ceil(shape_new[1]/fillwidth))
        nny = int(np.ceil(shape_new[0]/fillwidth))
        
        #Loop over the windows and calculate bounds for discrete part
        bounds = []
        for nx in range(nnx):
            for ny in range(nny):
                xmin = nx*fillwidth
                xmax = np.min(( (nx+1)*fillwidth, shape_new[1] ))
                ymin = ny*fillwidth
                ymax = np.min(( (ny+1)*fillwidth, shape_new[0] ))
                bounds.append([xmin, xmax, ymin, ymax])
                           
        #Fill the grid
        for b in bounds:
            grid_x, grid_y = np.meshgrid(np.arange(b[0],b[1]),
                                         np.arange(b[2],b[3]))
            interpd[b[2]:b[3],b[0]:b[1]] = interp(
                    np.vstack([grid_x.reshape(grid_x.size),
                grid_y.reshape(grid_y.size)]).T).reshape(grid_x.shape)
                        
    return interpd

#==============================================================================
        
def skymap(data, meshsize, medfiltsize=3, mask=None, method='cubic',
           skyfunc='median', rmsfunc='quantile', lowmem=False, tol=1.03, nits=6,
           getmask=False, fillfrac=0.3, ps=1, eps=5, kappa=5, thresh=0.5,
           verbose=False):
    '''
    Measure the sky and sky RMS in meshes, iteritively masking sources using
    DBSCAN.
    
    Parameters
    ----------
    data: 2D numpy array.
    
    meshsize (int): Background mesh unit size [pixels].
    
    medfiltsize (int): Median filter size [meshes].
    
    method (str): Interpolation type. Either 'cubic', 'linear' or 'nearest'.
    
    skyfunc (function or str): Function for sky estimation.
    
    rmsfunc (function or str): Function for sky RMS estimation.
         
    mask: 2D numpy array. Initial mask that grows in the function.
    
    lowmem (bool): Low memory mode?
    
    tol (float): Convergence tolerance for meshes.
    
    nits (int): Maximum number of masking iterations.
    
    getmask (bool): Return the mask?
    
    fillfrac (float): Minimum unmasked fraction in mesh before interpolation.
    
    Returns
    -------
    sky (2D numpy array): Interpolated sky map.
    
    rms (2D numpy array): Interpolated sky RMS map.
    '''
    
    if verbose: 
        print('skymap: measuring sky...')
        t0 = time.time()
    
    #Set up the mask
    if mask is None: 
        mask = np.zeros_like(data, dtype='bool')
    else:
        mask = mask>0
       
    #Make the mesh grid
    meshes, Nx, Ny = make_meshes(data.shape, meshsize, fillfrac=fillfrac)
    
    #Make the mesh array (required for median filtering)
    msky = np.zeros((Ny,Nx), dtype=data.dtype)
    mrms = np.zeros((Ny,Nx), dtype=data.dtype)
        
    #Do the incremental DBSCAN masking iterations
    count=0; finished=False
    while not finished:
        finished = True
        for mesh in [m for m in meshes if m.active]: #Loop over active meshes
        
            finished = False    #There are still active meshes
            
            #Calculate the mesh values
            msky[mesh.j,mesh.i], mrms[mesh.j,mesh.i] = mesh.measure(data,
                                  mask=mask, skyfunc=skyfunc, rmsfunc=rmsfunc)
                        
            #Check if the sky has converged
            if msky[mesh.j,mesh.i]/mesh.sky <= tol:
                mesh.deactivate()
                
            #Also terminate mesh if it is not finite (i.e. masked)
            elif (not np.isfinite(msky[mesh.j,mesh.i])):
                mesh.deactivate()
                
            #Update the mesh values
            mesh.update(msky[mesh.j,mesh.i], mrms[mesh.j,mesh.i])
            
        if not finished: #There are changes
                                   
            #Apply the median filter
            if medfiltsize > 1:
                msky = median_filter(msky, medfiltsize, mode='nearest')
                mrms = median_filter(mrms, medfiltsize, mode='nearest')
                                    
            #Fill in nans with nearest neighbours if they exist 
            cond_finite = np.isfinite(msky)
            if cond_finite.any():
                msky = fill_nans(msky)
            cond_finite = np.isfinite(mrms)
            if cond_finite.any():
                mrms = fill_nans(mrms)
                                                            
            #Interpolate the meshgrid to the full size (ignoring nans)
            sky = interpolate_meshgrid(msky,data.shape,lowmem=lowmem,method=method)
            rms = interpolate_meshgrid(mrms,data.shape,lowmem=lowmem,method=method)
                        
            #Use DBSCAN to update the mask
            if nits != 0:
                mask |= dbscan.dbscan(data, mask=None, sky=sky, eps=eps,
                                      kappa=kappa, thresh=thresh, ps=ps,
                                      rms=rms, verbose=False,
                                      ).segmap_dilate.astype('bool')
        #Check if max iterations have been reached     
        if count >= nits:
            if verbose: print('-WARNING: skymap reached max iterations.')
            finished = True
        if (nits!=0): count += 1
            
    #Return the maps
    if verbose: 
        meshes_converged = np.sum([(not m.active) for m in meshes])
        print('-Total iterations: %i.'%(count))
        print('-Final mesh convergence: %i%%.'%(100*meshes_converged/len(meshes)))
        print('skymap: finished after %i seconds.' % (time.time()-t0))
    if getmask:   
        return sky, rms, mask
    return sky, rms
            
    
