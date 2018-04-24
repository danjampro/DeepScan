#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:57:41 2017

@author: danjampro
"""

import os, subprocess, tempfile, time, multiprocessing
import numpy as np
import pandas as pd
from . import utils, geometry, sersic, NTHREADS

def sextract(data, mzero, ps, flux_frac=0.5, detect_thresh=1.5, 
             detect_minarea=5, deblend_mincont=0.005, clean=True, convolve=True,
             kernel=None, extras='', verbose=True, bgsize=64, sexpath='sex',
             segmap=False):
            
    t0 = time.time()
    
    #Create a temporary directory
    with tempfile.TemporaryDirectory() as dirpath:
        
        #Save the data as fits
        fitsdata = os.path.join(dirpath, 'data.fits')
        utils.save_to_fits(data, fitsdata)
        
        #Make the default configuration file
        config_name = os.path.join(dirpath, 'default.sex')
        with open(config_name, 'w') as config:
            subprocess.call([sexpath, "-d"], stdout=config)
            
        #Make the parameter file
        param_name = os.path.join(dirpath, 'default.param')
        with open(param_name, 'w') as param:
            param.write('X_IMAGE\nY_IMAGE\nMAG_AUTO\nFLUX_RADIUS\nA_IMAGE\n\
                        B_IMAGE\nTHETA_IMAGE\nMAG_ISO\nKRON_RADIUS\nFLUX_MAX\n\
                        ISOAREA_IMAGE\nXMAX_IMAGE\nXMIN_IMAGE\nYMAX_IMAGE\n\
                        YMIN_IMAGE\n')
            
        #Decide whether to save segimage
        if segmap:
            segname = os.path.join(dirpath, 'seg1.fits')
            extras = '%s -CHECKIMAGE_NAME %s -CHECKIMAGE_TYPE SEGMENTATION' % (extras, segname)
            
        #Convert the bools to Y or Ns
        if clean:
            clean = 'Y'
        else:
            clean = 'N'
        if convolve:
            convolve = 'Y'
            #Copy the convolution file
            convfile = os.path.join(dirpath, 'default.conv')
            if kernel is None:
                #Assume default filter
                write_filter(np.array([[1,2,1],[2,4,2],[1,2,1]]), convfile)
            else:
                write_filter(kernel, convfile)
                
        else:
            convolve = 'N'
            convfile = os.path.join(dirpath, 'default.conv')
            
        #Create a string to override default parameters 
        ofile = os.path.join(dirpath, 'output.cat')
        s = '-PHOT_FLUXFRAC %.3f -MAG_ZEROPOINT %.3f -PIXEL_SCALE %.3f -CATALOG_NAME %s -PARAMETERS_NAME %s -FILTER_NAME %s -DETECT_THRESH %s -DETECT_MINAREA %.5f -DEBLEND_MINCONT %i -CLEAN %s -FILTER %s -BACK_SIZE % i %s' % \
        (flux_frac, mzero, ps, ofile, param_name, convfile, detect_thresh,
         detect_minarea, deblend_mincont, clean, convolve, bgsize, extras)
        
        #Run SExtractor
        if verbose:
            print('-sextractor: sextracting...')
        subprocess.run('%s %s -c %s %s' % (sexpath, fitsdata, config_name, s),
                       check=True, shell=True, stdout=open(os.devnull, 'wb'),
                       stderr=open(os.devnull, 'wb'))

        t1 = time.time() - t0
        if verbose:
            print('-sextractor: finished after %i seconds.' % t1)
        
        #Return dataframe + segmap
        if segmap:
            segmap = utils.read_fits(segname)
            os.remove(segname)
            return read_output(ofile), segmap
        
        #Read the parameters into a pandas dataframe
        return read_output(ofile)
        

def write_filter(arr, fname, norm=False):
    
    '''Write filter arr to fname for SExtracting'''
    file = open(fname, 'w')
    if norm:
        arr /= arr.sum()
        file.write('CONV NONORM\n')
    else:
        file.write('CONV NORM\n')
    for row in arr.astype('str'):
        line = " ".join(row)
        file.write(line)
        file.write('\n')
    file.close()
    
    

def read_output(filename):
    keys = []
    with open(filename, 'r') as f:
        #Get the keys
        for line in f.readlines():
            if line[0] == '#':
                keys.append(line.split()[2])
    #Make container for data and fill
    data = [[] for key in keys]
    with open(filename, 'r') as f:
        #Get the keys
        for line in f.readlines():
            if line[0] != '#':
                split = line.split()
                for i in range(len(split)):
                    data[i].append(split[i])
    #Make pandas DF
    df = pd.DataFrame()
    for i in range(len(keys)):
        df[keys[i]] = np.array(data[i]).astype('float')
    return df



def ellipses_re(df, radius_key='FLUX_RADIUS'):
    ells = [geometry.ellipse(a=df[radius_key][i], 
                             b=(df['B_IMAGE'][i]/df['A_IMAGE'][i])*df[radius_key][i],
                             theta=(df['THETA_IMAGE'][i])*(np.pi/180)+np.pi/2,
                             x0=df['X_IMAGE'][i],
                             y0=df['Y_IMAGE'][i]) for i in range(df.shape[0])]
    return ells



class sextractor_ellipse(geometry.ellipse):
    '''Child class of ellipse that allows for extra information.'''
    def __init__(self, x0, y0, a, b, theta, flux_max=None):
        geometry.ellipse.__init__(self,x0=x0,y0=y0,a=a,b=b,theta=theta)
        self.flux_max = flux_max
        
def ellipses_iso(df, uiso, ps, Nthreads=NTHREADS):
    
    pool = multiprocessing.Pool(Nthreads)
    
    try:
    
        thetas = df['THETA_IMAGE'].as_matrix()*(np.pi/180)+np.pi/2
        res = df['FLUX_RADIUS'].as_matrix() * ps
        rks = df['KRON_RADIUS'].as_matrix() * ps
        
        ns = np.array(pool.starmap(sersic.index_re_kron,
                                   ((res[i], rks[i]) for i in range(res.size))))
        
        bs = np.array( pool.map(sersic.sersic_b, ns) )
        ues = sersic.effective_SB(df['MAG_AUTO'], res, ns, bs)
        qs = df['B_IMAGE']/df['A_IMAGE']
        risos = sersic.isophotal_radius(uiso, ues, res, ns, bs)
        
        ells = [sextractor_ellipse(a=risos[i] / ps, 
                                 b=risos[i]*qs[i] / ps,
                                 theta=thetas[i],
                                 x0=df['X_IMAGE'][i],
                                 y0=df['Y_IMAGE'][i],
                                 flux_max=df['FLUX_MAX'][i])
                                for i in range(res.size)]                              
    finally:
        pool.close()
        pool.join()
    
    return ells       



def get_ellipses(data, mzero, ps, uiso, mask=None, fillval=0, verbose=True,
                 Nthreads=NTHREADS, debug=True, **kwargs):
    '''
    Use SExtractor to create ellipses corresponding to isophotal radii.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    t0 = time.time()
    
    print('get_ellipses: running sextractor...')
    
    #Apply the mask
    if mask is not None:
        dat = data.copy()
        dat[mask] = fillval
    else:
        dat = data
        
    #Run SExtractor
    dfsex = sextract(dat, mzero=mzero, ps=ps, verbose=verbose, **kwargs)
    
    print('get_ellipses: estimating aperture sizes...')
    
    #Calculate ellipse size
    ellipses = ellipses_iso(dfsex, uiso, ps, Nthreads=Nthreads)
    
    #Clean result of nan values
    keepers = np.isfinite([e.a for e in ellipses])
    keepers *= np.array([e.a > 0 for e in ellipses],dtype='bool')
    keepers *= dfsex['MAG_AUTO'].as_matrix() < 99
    
    dfsex2 = pd.DataFrame()
    for col in dfsex.columns.values:
        dfsex2[col] = dfsex[col].as_matrix()[keepers]
    ellipses = [e for i, e in enumerate(ellipses) if keepers[i]]
    
    t1 = time.time() - t0
    print('get_ellipses: finished after %i seconds.' % t1)
    
    if debug:
        return ellipses, dfsex2
    return ellipses







    
    
      