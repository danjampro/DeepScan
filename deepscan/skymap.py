# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.interpolate import griddata
from . import dbscan
from .cython.cy_skymap import measure_sky, apply_median_filter, fill_nans, Mesh

#==============================================================================

def make_meshes(shape, meshsize, sky=0, rms=0):
    '''
    
    '''
    meshsize = int(np.ceil(meshsize))
    
    #Get mesh boundaries in pixel coordinates
    xmins = np.arange(0, shape[1], meshsize)
    xmaxs = xmins + meshsize; xmaxs[xmaxs>shape[1]] = shape[1]
    ymins = np.arange(0, shape[0], meshsize)
    ymaxs = ymins + meshsize; ymaxs[ymaxs>shape[0]] = shape[0]
    
    #Offset mesh if out of bounds
    for x in range(xmins.size):
        dx = shape[1] - xmaxs[x]
        if dx < 0:
            xmins[x] += dx
            xmaxs[x] += dx
            
    for y in range(ymins.size):
        dy = shape[0] - ymaxs[y]
        if dy < 0:
            ymins[y] += dy
            ymaxs[y] += dy
                            
    #Create the mesh grid
    meshes = np.empty((ymins.size, xmins.size), dtype=object)
    for x, (xmin, xmax) in enumerate(zip(xmins, xmaxs)):
        for y, (ymin, ymax) in enumerate(zip(ymins, ymaxs)):
            slc = slice(ymin, ymax), slice(xmin, xmax)
            meshes[y, x] = Mesh(x=x, y=y, slc=slc, sky=sky, rms=rms) 
    
    meshes = meshes.reshape(-1)
    
    return meshes, ymins.size, xmins.size

#==============================================================================

def make_mask(data, sky, rms, mask=None, use_fft=False, **kwargs):
    '''
    
    '''
    return dbscan.DBSCAN(data-sky, rms=rms, verbose=False, get_sources=False,
                         erode=False, label_segments=False, use_fft=use_fft,
                         mask=mask, **kwargs).segmap_dilate > 0
                         
#==============================================================================


def skymap(data, meshsize=100, nits=2, kappa=4, eps=5, thresh=0.5, 
           medfiltsize=3, interpolate=True, verbose=False, mask=None,
           use_fft=False, getmask=False, fillfrac=0.3, interp_kwargs={}):
    '''
    Measure the sky and sky RMS in meshes, iteritively masking sources using
    DBSCAN.
    
    Parameters
    ----------
    data: 2D float array
        The data array.
    
    meshsize : int
        Background mesh unit size [pixels].
    
    medfiltsize : int
        Median filter size [meshes].
                 
    mask : 2D bool array
        Optional initial mask.
        
    nits : int
        Number of DBSCAN iterations.
        
    fillfrac : float
        Minimum unmasked fraction in mesh before it is interpolated over.
        
    getmask : bool
        Also return the mask used for the sky estimate?
        
    interpolate : bool
        Interpolate meshes for final skymap?
        
    interp_kwargs : dict
        Keyword arguments to be passed to scipy.interpolate.griddata.
    
    Returns
    -------
    2D float array
        Interpolated sky image.
    
    2D float array
        Interpolated sky RMS image.     
    '''
    if verbose: 
        print('skymap: measuring sky...')
        t0 = time.time()
        
    if (medfiltsize==0) or (medfiltsize is None):
        medianfilter = False
    else:
        if medfiltsize%2 == 0:
            raise ValueError('medfiltsize must be odd.')
        dmed = int( (medfiltsize-1)/2 )        
        medianfilter = dmed != 0
        
    #Create the Mesh objects
    meshes, ny_, nx_ = make_meshes(data.shape, meshsize=meshsize)
    
    #Set up arrays    
    sky = np.empty(data.shape, dtype=data.dtype)
    rms = np.empty(data.shape, dtype=data.dtype)
    if mask is not None:
        mask = mask > 0
        mask2 = mask.copy()
    else:
        mask = ~np.isfinite(data)
        mask2 = mask.copy()
                        
    for it in range(nits+1):
                
        #Increment the mask
        if it != 0:
            mask2[:,:] = make_mask(data, sky, rms, kappa=kappa, eps=eps,
                                   thresh=thresh, mask=None, use_fft=use_fft)
            
            mask2 |= np.isnan(sky) #Not needed after fill nans implementation
            if mask is not None:
                mask2 |= mask #Always apply original mask
                                            
        #Do the sky estimate, updating the meshes
        measure_sky(data, mask=mask2.astype(np.uint8), meshes=meshes,
                    fillfrac=fillfrac)
                                                
        #Apply the median filter if necessary
        if medianfilter:
            apply_median_filter(meshes, nx_, ny_, dmed)
            
        #Update the mesh grids
        if medianfilter:
            for mesh in meshes:   
                mesh.sky = mesh.sky_
                mesh.rms = mesh.rms_
                                                            
        #Fill the nans
        if not (np.isfinite([m.sky for m in meshes]).any()) or not (
                np.isfinite([m.sky for m in meshes]).any()):
                    raise RuntimeError("No finite meshes! Modify settings.")
        else:
            fill_nans(meshes, ny_, nx_)
            pass
                    
        #Fill the full sized arrays
        for mesh in meshes:
            sky[mesh.slc] = mesh.sky
            rms[mesh.slc] = mesh.rms
                                    
    #Use an interpolated sky grid?
    if interpolate:
        
        if verbose:
            print('-Performing interpolation.')
        
        points = []
        sky_values = []
        rms_values = []
        
        for mesh in meshes:
            points.append([mesh.x+0.5, mesh.y+0.5])
            sky_values.append(mesh.sky)
            rms_values.append(mesh.rms)
            
            #Need to have points on the image boundaries
            if (mesh.x == 0):
                if (mesh.y == 0):
                    points.append([0, 0])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms)
                elif (mesh.y == ny_-1):
                    points.append([0, ny_])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms)
                else:
                    points.append([0, mesh.y+0.5])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms)                                   
            elif (mesh.x == nx_-1):
                if (mesh.y == 0):
                    points.append([nx_, 0])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms)
                elif (mesh.y == ny_-1):
                    points.append([nx_, ny_])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms) 
                else:
                    points.append([nx_, mesh.y+0.5])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms) 
            if (mesh.y == 0):
                if not mesh.x in [nx_-1, 0]:
                    points.append([mesh.x+0.5, 0])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms) 
            elif (mesh.y == ny_-1):
                if not mesh.x in [nx_-1, 0]:
                    points.append([mesh.x+0.5, ny_])
                    sky_values.append(mesh.sky); rms_values.append(mesh.rms)
           
        #Perform the interpolation
        points = np.vstack(points)
        xx, yy = np.meshgrid(np.linspace(0, nx_, data.shape[1]),
                             np.linspace(0, ny_, data.shape[0]))
        sky[:,:] = griddata(points, sky_values,(xx.reshape(-1),yy.reshape(-1)),
                       **interp_kwargs).reshape(data.shape) 
        rms[:,:] = griddata(points, rms_values,(xx.reshape(-1),yy.reshape(-1)),
                       **interp_kwargs).reshape(data.shape) 
        
    if verbose: 
        print('skymap: finished after %i seconds.' % (time.time()-t0))
                
    if getmask:
        return sky, rms, mask2
    return sky, rms

#==============================================================================
#==============================================================================
        
    
