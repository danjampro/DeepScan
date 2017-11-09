#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:23:33 2017

@author: Dan
"""

#import os, sys, subprocess, tempfile
#import time as Time
#from . import NTHREADS, geometry, utils, clusters, dbscan_conv
from . import dbscan_conv

def dbscan(*args, **kwargs):
    return dbscan_conv.dbscan_conv(*args, **kwargs)

"""
OLD STUFF

def call(fitsdata, 
         fitsnoise, 
         ofile, 
         thresh_min,  
         eps, 
         kappa=5, 
         minpts=0, 
         Nthreads=NTHREADS, 
         Nstrips=None, 
         ex=DBSCAN_EX, 
         verbose=False, 
         ps=None, 
         mzero=None,
         thresh_max = None,
         thresh_type='ADU',
         eps_unit='pix',
         n_type = 'RMS',
         o_type = 'ELLIPSE',
         ext_data=0,
         ext_noise=0,
         time=False,
         include_secondaries=False):
    
    '''Call external dbscan executable'''
    
    #Make sure executable is real & executable
    try:
        assert(os.path.isfile(ex))
    except:
        from . import DBSCAN_REPO
        raise( FileNotFoundError('Could not find DBSCAN executable: %s. If it is not installed, download it from %s.' % (ex, DBSCAN_REPO)) )
    try:
        assert(os.access(ex, os.X_OK))
    except:
        raise( OSError('Could not execute DBSCAN executable: %s' % ex) )
        
    assert(type(include_secondaries)==bool)
    
    if Nstrips is None:
        Nstrips=Nthreads
                
    args= ['%s' % ex,
           '--FITSDATA', fitsdata,
           '--FITSNOISE',  fitsnoise,
           '--OFILE', ofile,
           '--TMIN', '%s' % '{}'.format(thresh_min),
           '--NSTRIPS', '%i' % Nstrips,
           '--NTHREADS', '%i' % Nthreads,
           '--EPS', '%s' % '{}'.format(eps),
           '--MINPTS', '%s' % '{}'.format(minpts),
           '--CONFIDENCE', '%s' % '{}'.format(kappa),
           '--PS', '%s' % '{}'.format(ps),
           '--MZERO', '%s' % '{}'.format(mzero),
           '--THRESH_TYPE',  '{}'.format(thresh_type),
           '--EPS_UNIT',  '{}'.format(eps_unit),
           '--N_TYPE',  '{}'.format(n_type),
           '--O_TYPE',  '{}'.format(o_type),
           '--EXT_DATA',  '{}'.format(ext_data),
           '--EXT_NOISE',  '{}'.format(ext_noise),
           '--INCLUDE_SECONDARY_POINTS',  '{}'.format(int(include_secondaries))
           ]
    
    if thresh_max is not None:
        args.append('--TMAX', '%s' % '{}'.format(thresh_max))
        
    if verbose:
        args.append('-v')
        
    #Call the DBSCAN executable and check the return code
    t0 = Time.time()
    
    if sys.version_info[0] < 3:
        subprocess.check_call(args, shell=False)
    else:
        subprocess.run(args, shell=False, check=True)   #Arg validation is done in C++
        
    t1 = Time.time() - t0
    
    if o_type == 'ELLIPSE':
        #Return ellipses
        ellipses = read_ellipses(ofile)
        
        if time:
            return ellipses, t1
        else:
            return ellipses
        
    elif o_type == 'RAW':
        #Return raw clusters
        cls = clusters.read_all(ofile)
        if time:
            return cls, time
        else:
            return cls
    

def _dbscan(
           thresh_min,
           eps, 
           data=None, 
           noise=None, 
           fitsdata=None, 
           fitsnoise=None, 
           ofile=None,
           kappa=5, 
           minpts=0, 
           ps=1, 
           mzero=0, 
           Nthreads=NTHREADS, 
           Nstrips=None, 
           verbose=False, 
           overwrite=False, 
           thresh_max=None,
           n_type = 'RMS',
           ext_data = 0,
           ext_noise = 0,
           ex=DBSCAN_EX,
           time=False,
           eps_unit='ARCSEC',
           thresh_type='SB',
           o_type='ELLIPSE',
           include_secondaries=False):
    
    '''Do the unit conversions and call DBSCAN'''
    
    #Ensure correct data/noise input
    if data is not None:
        if fitsdata is None:
            #raise ValueError('fitsdata must be specified if data is not None type.')
            tempdata = tempfile.NamedTemporaryFile()
            fitsdata = tempdata.name
            try:
                utils.save_to_fits(data=data, fname=fitsdata, overwrite=overwrite)
            except:
                tempdata.close()
                raise #Reraise original error
        else:
            utils.save_to_fits(data=data, fname=fitsdata, overwrite=overwrite)
            tempdata = None
    else:
        tempdata = None
            
    if noise is not None:
        if fitsnoise is None:
            #raise ValueError('fitsnoise must be specified if noise is not None type.')
            tempnoise = tempfile.NamedTemporaryFile()
            fitsnoise = tempnoise.name
            try:
                utils.save_to_fits(data=noise, fname=fitsnoise, overwrite=overwrite)
            except:
                tempnoise.close()
                raise
        else:
            utils.save_to_fits(data=noise, fname=fitsnoise, overwrite=overwrite)
            tempnoise = None
    else:
        tempnoise = None
            
    if ofile is None:
        tempout = tempfile.NamedTemporaryFile()
        ofile = tempout.name
    else:
        tempout = None
                
    try:               
        #Call DBSCAN
        output = call(fitsdata=str(fitsdata),
                        fitsnoise=str(fitsnoise),
                        ofile=str(ofile),
                        thresh_min=thresh_min,
                        thresh_max=thresh_max,
                        minpts=minpts,
                        kappa=kappa,
                        eps=eps,
                        Nstrips=Nstrips,
                        Nthreads=Nthreads,
                        ex=ex,
                        verbose=verbose,
                        ps=ps,
                        mzero=mzero,
                        thresh_type=thresh_type,
                        eps_unit=eps_unit,
                        n_type=n_type,
                        o_type=o_type,
                        ext_data = ext_data,
                        ext_noise = ext_noise,
                        time=time,
                        include_secondaries=include_secondaries
                        )
    #Close and delete temp files
    finally:
        if tempnoise:
            tempnoise.close()
        if tempdata:
            tempdata.close()
        if tempout:
            tempout.close()
    
    return output
    

           
def read_ellipses(cfile, radius_type='shape_max'):
    
    '''Read ellipses from cfile'''
    
    import pandas as pd
    
    #Check for file
    try:
        assert( os.path.isfile(cfile) )
    except:
        raise( FileNotFoundError('Could not find cluster file: %s' % cfile) )
        
    if radius_type == 'shape_max':
        a_key = 'a_shape_max'
        b_key = 'b_shape_max'
    elif radius_type == 'shape_rms':
        a_key = 'a_shape_rms'
        b_key = 'b_shape_rms'
    elif radius_type == 'rms':
        a_key = 'a_rms'
        b_key = 'b_rms'
    else:
        raise KeyError('radius_type does not exist: %s' % radius_type)
    
    #Attempt to read ellipses from cluster file
    try:
        DF = pd.read_csv(cfile)
        ellipses = [ geometry.ellipse(x0=DF['x0'][i],
                                    y0=DF['y0'][i],
                                    a=DF[a_key][i],
                                    b=DF[b_key][i],
                                    theta=DF['theta'][i])
                    for i in range(DF.shape[0]) ]     
    except:
        raise( ValueError('Could not read cluster file %s' % cfile) )
        
    return ellipses

"""

