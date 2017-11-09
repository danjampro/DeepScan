#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:59:19 2017

@author: Dan
"""

import numpy as np
from . import dbscan

def match_ellipse(bands):
    
    '''Output all ellipses from bands[0] that have detections across all bands'''
    
    Nbands = len(bands)
    idxs = []
    output = []
    
    #Loop over ellipses in first band
    for i, e0 in enumerate(bands[0]):
        
        for band in bands[1:]:
            
            #Loop over ellipses in band n
            for e in band:
                
                #If match, save idx of band 0
                if e0.check_inside(e.x0, e.y0):
                    
                    idxs.append(i)
                    
                    #End search of this band - there could be duplicates
                    continue 
                    
    #Now must check ellipses were detected across all bands
    idxs = np.array(idxs)
    for uid in np.unique(idxs):
        
        #Count number of occurances, save if equal to number of extra bands
        if np.sum(idxs==uid) == Nbands-1:
            output.append(bands[0][uid])
            
    return output


def main(cfile_list, ofile, overwrite=True, verbose=False):
    
    '''Save ellipses from cfile_list[0] that have at least one match in all other cfiles'''
    
    import os
    import pandas as pd
                                    
    #Read ellipses from cfiles
    bands = []
    for cfile in cfile_list:
        ellipses = dbscan.read_ellipses(cfile)
        bands.append(ellipses)
        
    #Do the matching
    matches = match_ellipse(bands)
    Nmatches = len(matches)
    
    if verbose:
        print('match: %i matches' % Nmatches)
    
    #Save to csv
    if overwrite == False:
        if os.path.isfile(ofile):
            raise(IOError('File %s exists. Set overwrite=True to overwrite' % ofile))
    
    x0s = np.zeros(Nmatches, dtype='float')
    y0s = np.zeros(Nmatches, dtype='float')
    aas = np.zeros(Nmatches, dtype='float')
    bbs = np.zeros(Nmatches, dtype='float')
    qqs = np.zeros(Nmatches, dtype='float')
    thetas = np.zeros(Nmatches, dtype='float')
    
    for i, E in enumerate(matches):
        x0s[i] = E.x0
        y0s[i] = E.y0
        aas[i] = E.a
        bbs[i] = E.b
        qqs[i] = E.q
        thetas[i] = E.theta
    DF = pd.DataFrame()
    DF['x0'] = x0s
    DF['y0'] = y0s
    DF['a'] = aas
    DF['b'] = bbs
    DF['q'] = qqs
    DF['theta'] = thetas
    DF.to_csv(ofile)
         
        
    
    
        
                    
                    
                    
            
            

