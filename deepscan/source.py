#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:11:02 2019

@author: danjampro

The Source class, to help generate catalogues from the data and a segmap.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects

#==============================================================================
#Source class

class Source():

    def __init__(self, segID, slc):
        '''
        Parameters
        ----------
        segID : int
            The unique segment ID.
            
        slc : tuple of two slices.
            The slice identifying the source.        
        '''
        self.segID = segID
        self.slc = slc
        self.sizex = slc[1].stop-slc[1].start
        self.sizey = slc[0].stop-slc[0].start
        self.series = pd.Series({'segID':segID, 'xmin':slc[1].start,
                                 'xmax':slc[1].stop, 'ymin':slc[0].start,
                                 'ymax':slc[0].stop})
        
        
    def get_data(self, data, segmap):
        '''
        Get the data for the source using the segmap.
        
        Parameters
        ----------
        data : 2D float array
            The data array.
        
        segmap : 2D int array
            The segmentation image.
        
        Returns
        -------
        1D float array
            The data for the source.
        '''
        return data[self.slc][segmap[self.slc]==self.segID]
        
        
    def get_coordinates(self, segmap):
        '''
        Get the coordinates of the segment in the original image.
        
        Parameters
        ----------
        segmap : 2D int array
            The segmentation image.
            
        Returns
        -------
        1D float array
            x coordinates.
        
        1D float array
            y coordinates.
        '''
        xx, yy = np.meshgrid(np.arange(self.sizex), np.arange(self.sizey))
        xx = xx[segmap[self.slc]==self.segID] + self.slc[1].start
        yy = yy[segmap[self.slc]==self.segID] + self.slc[0].start  
        return xx, yy
        
    
    def add_measurements(self, series):
        '''
        Add measurements to self.series.
        
        Parameters
        ----------
        series : pandas.Series
            The series to be merged. Duplicate keys are overwritten.
        '''
        for key in series.keys():
            self.series[key] = series[key]
    
#==============================================================================

def get_sources(segmap):
    '''
    Generate sources from the segmap.
    
    Parameters
    ----------
    segmap : 2D int array
        The segmentation image.
        
    Returns
    -------
    list of deepscan.source.Source
        The sources for each segment.
    '''
    return [Source(_1+1,_2) for _1, _2 in enumerate(
                                    find_objects(segmap)) if _2 is not None]
    
#==============================================================================
#==============================================================================