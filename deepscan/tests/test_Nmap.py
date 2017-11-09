#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:26:37 2017

@author: Dan
"""

import sys; sys.path.append('../../') #Get rid of.

if __name__ == '__main__':
    
    import numpy as np
    from deepscan import Nmap
    import matplotlib.pyplot as plt; plt.close('all')
    
    #Settings
    dshape = (500, 500)
    WSIZE = 13
    SIG = 3
    THIN = 2
    BSIZE = 100 
    
    RMS1 = 10
    RMS2 = 20
    
    #Coordinates for masking
    mx1=450
    mx2=550
    
    #Make the RMS map
    Rx = int(dshape[1]/2)
    Ry = int(dshape[0]/2)
    RMS = np.zeros(dshape)
    RMS[0:Ry, 0:Rx] = RMS1
    RMS[Ry:2*Ry, Rx:2*Rx] = RMS1
    RMS[Ry:2*Ry, 0:Rx] = RMS2
    RMS[0:Ry, Rx:2*Rx] = RMS2
    RMS[mx1-BSIZE:mx2-BSIZE,mx1-BSIZE:mx2-BSIZE] = 50 #Region to be masked
    
    #Make a realisation of the RMS map
    noise = np.random.normal(loc=0, scale=RMS, size=dshape)
    
    #Embed data in a 0-boarder
    data = np.zeros((dshape[0]+2*BSIZE, dshape[1]+2*BSIZE))
    data[BSIZE:-BSIZE, BSIZE:-BSIZE] = noise
    
    RMS_ = np.zeros((dshape[0]+2*BSIZE, dshape[1]+2*BSIZE))
    RMS_[BSIZE:-BSIZE, BSIZE:-BSIZE] = RMS
    RMS = RMS_
    
    #Create a mask
    
    mask = np.zeros_like(data)
    mask[mx1:mx2, mx1:mx2] = 1
    
    #Use Nmap to measure the RMS from the realisation
    bg, sdata = Nmap.Nmap(data, wsize=WSIZE, sig=SIG, thinning_factor=THIN, mask=mask)
    
    #Calculate the residual (ignoring mask)
    RMS[mx1:mx2, mx1:mx2] = 10
    residual = RMS - sdata
    RMS[mx1:mx2, mx1:mx2] = 50
    
    #Display
    fig = plt.figure()
    
    ax0 = fig.add_subplot(2,2,1)
    im0 = ax0.imshow(RMS, cmap='binary')
    fig.colorbar(im0, ax=ax0)
    plt.title('True RMS')
    
    plt.plot([mx1,mx1],[mx1,mx2], 'r-')
    plt.plot([mx2,mx2],[mx1,mx2], 'r-')
    plt.plot([mx1,mx2],[mx2,mx2], 'r-')
    plt.plot([mx1,mx2],[mx1,mx1], 'r-')
    
    plt.plot([BSIZE,BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([dshape[1]+BSIZE,dshape[1]+BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[dshape[0]+BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[BSIZE,BSIZE], 'b-')
        
    ax1 = fig.add_subplot(2,2,2)
    im1 = ax1.imshow(data, cmap='binary')
    fig.colorbar(im1, ax=ax1)
    plt.plot([BSIZE,BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([dshape[1]+BSIZE,dshape[1]+BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[dshape[0]+BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[BSIZE,BSIZE], 'b-')
    plt.title('RMS Realisation')
    
    ax2 = fig.add_subplot(2,2,3)
    im2 = ax2.imshow(sdata, cmap='binary')
    fig.colorbar(im2, ax=ax2)
    plt.plot([BSIZE,BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([dshape[1]+BSIZE,dshape[1]+BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[dshape[0]+BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[BSIZE,BSIZE], 'b-')
    plt.title('Measured RMS')
    
    ax3 = fig.add_subplot(2,2,4)
    im3 = ax3.imshow(abs(residual), cmap='binary')
    fig.colorbar(im3, ax=ax3)
    plt.plot([BSIZE,BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([dshape[1]+BSIZE,dshape[1]+BSIZE],[BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[dshape[0]+BSIZE,dshape[0]+BSIZE], 'b-')
    plt.plot([BSIZE,dshape[1]+BSIZE],[BSIZE,BSIZE], 'b-')
    plt.title('|True - Measured RMS|')
    
    plt.tight_layout()
    
    #plt.savefig('/Users/danjampro/Dropbox/phd/images/test_Nmap.png', dpi=150, bbox_inches='tight')
    