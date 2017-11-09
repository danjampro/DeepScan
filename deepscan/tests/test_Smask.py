#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 17:22:47 2017

@author: Dan
"""

import sys; sys.path.append('../../') #Get rid of.

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    from deepscan import Smask, geometry, NTHREADS, BUFFSIZE
    
    plt.close('all')
    
    #Settings
    dshape = (500, 500)
    N_ellipses = 13
    Nprocs = NTHREADS
    buffsize = BUFFSIZE
 
    #Generate data
    data = np.zeros(dshape)
    noise = np.zeros(dshape) + 10
    
    #Make some arbitrary ellipses to mask
    Es = []
    for i in range(N_ellipses):
        x0 = np.random.uniform(low=0, high=dshape[1])
        y0 = np.random.uniform(low=0, high=dshape[0])
        a = np.random.uniform(low=10, high=30)
        q = np.random.uniform(low=0.3, high=1.)
        theta = np.random.uniform(-np.pi, np.pi)
        Es.append( geometry.ellipse(x0=x0, y0=y0, a=a, b=q*a, theta=theta) )
    
    
    #Do the masking
    masked = Smask.source_mask(data, Es,  noise, Nprocs=4, buffsize=buffsize)
    
    #Show the result
    fig, ax = plt.subplots()
    ax.imshow(masked, cmap='binary')
    [e.draw(ax, color='r') for e in Es]
    for e in Es:
        e.a
        e.b
        e.draw(ax, color='b')
    plt.xlim(0, dshape[1])
    plt.ylim(0, dshape[0])



