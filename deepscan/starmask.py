#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:18:26 2017

@author: danjampro

Issue of errors in find_centres when minarea is low and there are small detections.
"""

import numpy as np
from . import geometry, Smask, NTHREADS
from scipy.ndimage.measurements import label, labeled_comprehension
from scipy.ndimage.filters import maximum_filter
import os, functools, tempfile, shutil
from multiprocessing import Pool


def process_init(data, output_memmap):
    '''Initialiser for process pool'''
    global data_arr, output_arr
    data_arr = data
    output_arr = output_memmap
    
    
    
def find_centres(data, sat, rfactor=1, minarea=0, pixel_step=1, maxfiltershape=None, Nthreads=NTHREADS, labeled=None):
    '''Find the centres of saturated diffraction crosses'''
        
    if maxfiltershape is None:
        msk = data > sat
    else:
        msk = maximum_filter(data, size=maxfiltershape) > sat
    
    #Get unique regions
    print('getting labels')
    if labeled is None:
        labeled, N = label(msk)
    else:
        N = int(labeled.max())
    print('got the labels')
        
    #Get the bounding rectangles for each detection
    print('getting coords')
    indices = np.arange(1,N+1)
    xx, yy = np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))
    xmins = labeled_comprehension(input=xx, labels=labeled, index=indices,
                                   func=np.min, out_dtype='int', default=0)
    xmaxs = labeled_comprehension(input=xx, labels=labeled, index=indices,
                                   func=np.max, out_dtype='int', default=0)
    ymins = labeled_comprehension(input=yy, labels=labeled, index=indices,
                                   func=np.min, out_dtype='int', default=0)
    ymaxs = labeled_comprehension(input=yy, labels=labeled, index=indices,
                                   func=np.max, out_dtype='int', default=0)
    x0s = labeled_comprehension(input=xx, labels=labeled, index=indices,
                                   func=lambda X: np.round(np.mean(X)), out_dtype='int', default=0)
    print('got coords')
    
    #print(np.max(labeled), xmins, xmaxs, ymins, ymaxs, xmins.shape)
    
    #Calculate bounds for threads
    idxs_min = np.linspace(1, N, Nthreads+1)[:-1].astype('int')
    idxs_max = np.zeros_like(idxs_min); idxs_max[:-1]=idxs_min[1:]; idxs_max[-1]=N
    
    #Make memory maps for data and label
    dirpath = tempfile.mkdtemp()
    pool = None
    try:
        
        #Make temporary file names
        datafile  = os.path.join(dirpath, 'data.dat')
        #labelfile = os.path.join(dirpath, 'label.dat')
    
        #Put data into arrays
        fp_data  = np.memmap(datafile, dtype='float32', mode='w+', shape=data.shape)
        fp_data[:]  = data[:]
        #fp_label = np.memmap(labelfile, dtype='int', mode='w+', shape=data.shape)
        #fp_label[:] = labeled[:]
        
        #Flush the filled arrays to disk
        del fp_data
        #del fp_label
        
        #Loop over threads
        partial_func = functools.partial(find_centres_threaded,
                                         datafile=datafile, 
                                         N=N,
                                         rfactor=rfactor,
                                         pixel_step=pixel_step,
                                         minarea=minarea,
                                         x0s=x0s,
                                         xmaxs=xmaxs,
                                         xmins=xmins,
                                         ymins=ymins,
                                         ymaxs=ymaxs,
                                         dshape=data.shape)
        
        #Create a thread pool & run 
        p = Pool(Nthreads)
        result = p.starmap(partial_func, [(idxs_min[i],idxs_max[i]) for i in range(Nthreads)])
        
        #Extract info
        crds  = []
        boxes = []
        for thread in range(Nthreads):
            crds.extend(result[thread][0])
            boxes.extend(result[thread][1])
            
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        shutil.rmtree(dirpath)
     
    return crds, boxes


def find_centres_threaded(label_min,
                          label_max,
                          datafile, 
                          N,
                          rfactor, 
                          minarea,
                          pixel_step,
                          x0s,
                          xmins,
                          xmaxs,
                          ymins,
                          ymaxs,
                          dshape,
                          ):
    crd_list = []
    box_list = []
    
    data = np.memmap(datafile, dtype='float32', mode='r+', shape=dshape)
    #labeled = np.memmap(labelfile, dtype='int', mode='r+', shape=dshape)
        
    for i in range(label_min,label_max):
        
        xmax = xmaxs[i-1]
        xmin = xmins[i-1]
        ymax = ymaxs[i-1]
        ymin = ymins[i-1]
        x0 = x0s[i-1]
        
        #Ignore small detections which are ususally associated with larger ones
        if (xmax-xmin)*(ymax-ymin) > minarea:
                    
            xradius = int( (xmax-xmin)/2 * rfactor )
            
            #Find xcoord that has largest row intensity over characteristic size
            ys = np.arange(ymin, ymax+pixel_step, step=pixel_step)
            sels = np.zeros_like(ys)
            
            #Loop over rows
            for j, y in enumerate(ys):
                sels[j] = np.sum( data[y:y+1, x0-xradius:x0+xradius] )
            y0 = ys[np.argmax(sels)]
            yradius = int( (ymax - ymin)/2 * rfactor )
            
            #Now stretch the y scale as this is typically an underestimate
            ymax = int(rfactor*(ymax-y0) + y0)
            ymin = int(rfactor*(ymin-y0) + y0)
            
            #Save the coordinate
            crd_list.append([x0, y0])
            
            #Create two boxes for the detection
            box_list.append(geometry.box(x0-xradius,x0+xradius,ymin,ymax,x0=x0,y0=y0))
            box_list.append(geometry.box(x0-yradius,x0+yradius,y0-xradius,y0+xradius))
        
        else:
            continue
     
    return crd_list, box_list



def estimate_radius(x0, y0, Imin, dr=1, R0=0, annulus_width=1,
                    countdown=1, Rmax=np.inf, Rfactor=1, mask=None):
    '''Find radius at which intensity drops to Imin'''
    
    #Increment radii in while loop
    done = False
    R = R0
    
    '''
    #Read data
    data = np.memmap(datafile, dtype='float32', mode='r+', shape=dshape)
    if maskfile is not None:
        mask = np.memmap(maskfile, dtype='float32', mode='r+', shape=dshape)
    '''
    
    #Need to have several concentric annuli under the brightness threshold to stop secondary rings
    countdown_now = countdown
    
    Rsaved = 0 #This is the final radius minus the annulus width
    while done is False:
        
        #Get boundaries
        xmin = int(x0-R-annulus_width)
        xmax = int(x0+R+annulus_width+1)
        ymin = int(y0-R-annulus_width)
        ymax = int(y0+R+annulus_width+1)
        
        #Crop boundaries
        xovr = np.max((data_arr.shape[1],xmax)) - data_arr.shape[1]
        xund = -np.min((0, xmin)) 
        yovr = np.max((data_arr.shape[0],ymax)) - data_arr.shape[0]
        yund = -np.min((0,ymin)) 
        
        xmin = xmin + xund
        ymin = ymin + yund
        xmax = xmax - xovr
        ymax = ymax - yovr
        
        #Select anular region
        xx, yy = np.meshgrid(np.arange(-R-annulus_width+xund, R+annulus_width+1-xovr),
                             np.arange(-R-annulus_width+yund, R+annulus_width+1-yovr))
        if mask is None:
            annulus = (xx**2+yy**2>R**2) * (xx**2+yy**2<(R+annulus_width)**2)
            Ienc = np.mean(data_arr[ymin:ymax,xmin:xmax][annulus])
        else:
            mask_ = ~data_arr[ymin:ymax,xmin:xmax].astype('bool')
            annulus = ((xx**2+yy**2>R**2) * (xx**2+yy**2<(R+annulus_width)**2))[mask_]
            
            Ienc = np.median(data_arr[ymin:ymax,xmin:xmax][mask_][annulus])
                
        #Check finished condition
        if Ienc <= Imin:
            
            countdown_now -= 1 
            
            #If first countdown, save radius
            if countdown_now == countdown-1:
                Rsaved = R

            #If countdown has finished, process is complete
            if countdown_now == 0:
                done = True
            else:
                R += dr
                
                #Have a maximum radius condition
                if R >= Rmax:
                    if Rsaved == 0:
                        Rsaved = Rmax
                    done = True
        else:
            countdown_now = countdown
            R += dr
            
    #Fill the memmap
    R = (Rsaved+annulus_width)*Rfactor
    xx, yy = np.meshgrid( np.arange(-R, R+1), np.arange(-R, R+1) )
    cond1 = xx**2 + yy**2 <= R**2
    xx += x0
    yy += y0
    cond2 = (xx<data_arr.shape[1]) * (xx>=0) * (yy<data_arr.shape[1]) * (yy>=0)
    cond = cond1*cond2
    output_arr[yy[cond], xx[cond]] = True
    

    #return geometry.ellipse(x0=x0,y0=y0,a=(Rsaved+annulus_width)*Rfactor,b=(Rsaved+annulus_width)*Rfactor,theta=0)


"""
def estimate_radius_threaded(idx_min, idx_max, crd_list,
                             box_list, Imin, dr, R0, countdown, annulus_width,
                             Rmax, Rfactor):
    '''Run estimate_radius for several sources'''
    ellipses = np.zeros(idx_max-idx_min, dtype='object')
    for i in range(idx_min, idx_max):
        ellipses[i-idx_min]=estimate_radius(crd_list[i][0], 
                                         crd_list[i][1],
                                         Imin=Imin,
                                         dr=dr,
                                         R0=R0,
                                         countdown=countdown,
                                         annulus_width=annulus_width,
                                         Rmax=Rmax,
                                         Rfactor=Rfactor)
    return ellipses
"""

def create_mask(data, sat, Imin, rfactor=1, minarea=0, dr=1, R0=0, 
                maxfiltershape=None, countdown=1, Nthreads=NTHREADS,
                pixel_step=1, annulus_width=1, Rmax=np.inf, Rfactor=1,
                memmap_mask=False):
    
    print('getting centres')
    
    #Get centres
    crds, boxes = find_centres(data, 
                               sat, 
                               rfactor=rfactor, 
                               minarea=minarea, 
                               maxfiltershape=maxfiltershape)
    
    
    
    #Create an output mask array stored as a memmap
    #with tempfile.TemporaryDirectory() as dirpath:
    temppath = tempfile.mkdtemp()
    pool = None
    
    try:
    

        #Put data into arrays
        fp_result  = np.memmap(os.path.join(temppath, 'result.dat'), dtype='bool', mode='w+', shape=data.shape)
        fp_result[:]  = np.zeros(data.shape, dtype='bool')

        print('filling boxes')
        
        #Fill the boxes
        for b in boxes:
            result = b.fill(fp_result, fillval=True)
                        
        #Get circles using result as a mask
        #Create a thread pool
        p = Pool(processes=Nthreads, initializer=process_init, initargs=(data, fp_result))
        
        #Define a partial function for estimate_radius_threaded
        partial_func = functools.partial(estimate_radius, 
                                         Imin=Imin,
                                         dr=dr,
                                         R0=R0,
                                         countdown=countdown,
                                         annulus_width=annulus_width,
                                         Rmax=Rmax,
                                         Rfactor=Rfactor)
        
        #Calculate bounds for threads
        N = len(crds)
        idxs_min = np.linspace(0, N+1, Nthreads+1)[:-1].astype('int')
        idxs_max = np.zeros_like(idxs_min); idxs_max[:-1]=idxs_min[1:]; idxs_max[-1]=N
        
        print('getting circles')
        
        #Initiate!
        #circles  = p.starmap(partial_func, [(idxs_min[i],idxs_max[i]) for i in range(Nthreads)])
        p.starmap(partial_func, [[crds[i][0], crds[i][1]] for i in range(N)])
                                     
        
        '''
        #circles = [estimate_radius(C[0], C[1], data, Imin, dr=dr, R0=R0,mask=result,countdown=countdown) for C in crds]
        circles  = np.hstack(circles)
            
        print('filling circles')
        
        
        #Fill the annuli
        result = Smask.source_mask(data=result,
                                         noise=None,
                                         fillval=1,
                                         cls=circles,
                                         Rfactor=1).astype('bool')
        '''
        result = np.array(fp_result)
        
        print('done.')
        
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        del fp_result
        shutil.rmtree(temppath)
            
    return result

    