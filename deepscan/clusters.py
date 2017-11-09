#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:40:28 2017

@author: Dan
"""

import os
import numpy as np
from scipy.ndimage.morphology import binary_dilation


class Cluster():
    
    def __init__(self, ID, fname, implicit_load=True):
        """
        Initialise Cluster.
        
        Parameters
        ----------
        ID : int, unique ID number of cluster
        
        fname : string, name of cluster file
        
        data_array : 2D float array, raw data
        
        implicit_load : bool, True if __init__ reads data
        """
        
        #Perform input check
        check = [type(fname) is str,
                 type(ID) is int,
                 type(implicit_load) is bool]
                 
        if ~np.array(check).all():
            raise IOError("Cluster parameters: ID(int), fname(string)")
        
        if implicit_load:
            if os.path.isfile(fname)==False:
                raise IOError("Missing cluster file: " + fname)
            
        #Class variables
        self.ID = ID
        self.Xs = []
        self.Ys = []
        self.band_data = {}
        
        if implicit_load == True:
            
            #Read data
            with open(fname) as f:
                for line in f:
                    data = line.split('\t') 
                    
                    #Select relevant data
                    if int(data[2].strip('\n')) == ID:
                        
                        try:
                            self.Ys.append(int(data[0]))
                            self.Xs.append(int(data[1]))
                        except:
                            raise IOError("Invalid cluster file")
            
            #List into array
            self.Xs = np.array(self.Xs)
            self.Ys = np.array(self.Ys)
            
            
    def add_coordinate(self, x, y):
        
        '''Add coordinate to cluster'''

        self.Xs.append(x)
        self.Ys.append(y)
        
        
    def dilate(self, kernel):
        
        '''Perform binary dilation and return dilated cluster'''
        
        fsize_x = int( np.ceil(kernel.shape[1]/2) )
        fsize_y = int( np.ceil(kernel.shape[0]/2) )
        xx, yy = np.meshgrid(np.arange(self.Xs.min()-fsize_x, self.Xs.max()+fsize_x+1),
                             np.arange(self.Ys.min()-fsize_y, self.Ys.max()+fsize_y+1))
        c2d = np.zeros_like(xx, dtype='int')
        for x, y in zip(self.Xs, self.Ys):
            c2d[(xx==x) * (yy==y)] = 1
        dilated = binary_dilation(c2d, kernel)
        Cnew = Cluster(0, 'None', False)
        for x, y in zip(self.Xs, self.Ys):
            if dilated[(xx==x) * (yy==y)] == 1:
                Cnew.add_coordinate(x, y)
        return Cnew
    
    

        
            

def read_all(cfile, minc=0, maxc=float('inf')):
    
    """ Create cluster objects efficiently for all clusters in file """
    
    clusters = []
    IDs = []
    
    #Check if file exists  
    import os; 
    if os.path.isfile(cfile)==False:
        raise IOError("Missing cluster file: " + cfile)
            
    idxID = 2
    idxY = 0
    idxX = 1
        
    #Read data
    IDold = -1
    IDloc = None
    
    ID_count = 0 #Counts number of clusters (used in literal list index mode)
    
    #Loop over lines in cluster file
    with open(cfile) as f:
        for line in f:
            
            #Split line string and read cluster ID
            data = line.split('\t') 
            ID = int(data[idxID].strip('\n'))
            
            #Don't read in clusters with IDs smaller than minc
            if ID < minc:
                continue
                
            #End if max cluster number has been read in (assumes ordered clusters)
            elif ID >= maxc:
                break
                            
            #If coordinate has not changed since last line
            if ID == IDold:         #No need to recheck location
                clusters[IDloc].Ys.append(int(data[idxY]))
                clusters[IDloc].Xs.append(int(data[idxX]))                
            
            #If cluster has already been made
            elif ID in IDs:         
                IDloc = np.argmax(np.array(IDs)==ID)
                clusters[IDloc].Ys.append(int(data[idxY]))
                clusters[IDloc].Xs.append(int(data[idxX]))
                
            #Make new cluster if it is a new ID
            else:      
                IDs.append(ID)
                clusters.append(Cluster(ID, cfile, implicit_load=False))
                IDloc = -1
                clusters[IDloc].Ys.append(int(data[idxY]))
                clusters[IDloc].Xs.append(int(data[idxX]))
                
            #Update IDold
            IDold = ID   
            
            #Update ID count
            ID_count += 1
    
    #List to array conversion
    for c in clusters:
        c.Xs = np.array(c.Xs)
        c.Ys = np.array(c.Ys)
        
    return clusters



def C2D(clusters, shape):
    grid = np.zeros(shape)
    for i, C in enumerate(clusters):
        for x, y in zip(C.Xs, C.Ys):
            grid[y,x] = i+1
    return grid