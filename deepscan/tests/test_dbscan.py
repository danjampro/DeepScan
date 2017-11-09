#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:06:50 2017

@author: Dan
"""

if __name__ == '__main__':
    
    import sys; sys.path.append('../../') #Get rid of.
    DBSCAN_EX = '/Users/Dan/Dropbox/phd/codes/scan/Debug/scan'
    
    import numpy as np
    import matplotlib.pyplot as plt
    from deepscan import ellipse, dbscan, utils
    import os, shutil, tempfile
    
    #Ensure the file is read/write by the creator only
    
    #Define some variables
    dshape = (500, 500)
    Ngals = 10
    
    #DBSCAN settings
    Imin = 0.9
    Imax = 1.1
    eps=10
    minpts=100
    Nthreads = 4
    Nstrips = 4
              
    #Make some data
    data = np.zeros(dshape)
    
    X, Y = np.meshgrid(np.arange(dshape[1]), np.arange(dshape[1]))
    
    for i in range(Ngals):
        a = np.random.uniform(low=10, high=50)
        q = np.random.uniform(low=0.3, high=1)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x0 = np.random.uniform(low=0, high=dshape[1])
        y0 = np.random.uniform(low=0, high=dshape[0])
        b = q * a
        
        e = ellipse.ellipse( x0=x0, y0=y0, a=a, b=b, theta=theta )
        inside = e.check_inside( X, Y )
        data[inside] = 1
     
    data += np.random.normal( loc=0, scale=0.1, size=dshape)
    
    #Make a temporary directory
    tmpdir = tempfile.mkdtemp()
    
    #Save data to fits
    tfits = os.path.join(tmpdir, './temp.fits')
    tcls = os.path.join(tmpdir, './temp.cls')
        
    try: 
        utils.save_to_fits(data, fname=tfits, overwrite=True)
    
        #Call dbscan
        ellipses = dbscan.call(fitsdata=tfits, 
                               fitsnoise='None',
                               ofile=tcls,
                               tmin=Imin,
                               tmax=Imax,
                               eps=eps,
                               minpts=minpts,
                               Nthreads=Nthreads,
                               Nstrips=Nstrips,
                               ex=DBSCAN_EX)
    finally:    
        #Clean up the temp stuff
        shutil.rmtree(tmpdir)
    
    #Make the plots
    plt.figure()
    plt.imshow(data, cmap='binary')
    [e.draw(plt.gca(), color='orange') for e in ellipses]
    plt.xlim(0, dshape[1])
    plt.ylim(0, dshape[0])