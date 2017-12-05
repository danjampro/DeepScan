#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:02:27 2017

@author: danjampro

"""

import os, tempfile, multiprocessing, shutil
import numpy as np
from scipy.signal import fftconvolve
from astropy.convolution import Tophat2DKernel
from functools import partial
from scipy.ndimage.measurements import label, find_objects
from . import geometry, BUFFSIZE
import time

#==============================================================================

def fit_ellipse(xs,ys,weights=None,rms=False):
    
    #First order moments
    x0 = np.average(xs,weights=weights)
    y0 = np.average(ys,weights=weights)
    
    #Second order moments
    x2 = np.average(xs**2,weights=weights) - x0**2
    y2 = np.average(ys**2,weights=weights) - y0**2
    xy = np.average(xs*ys,weights=weights) - x0*y0
    
    #Handle infinitely thin detections
    if x2*y2 - xy**2 < 1./144:
        x2 += 1./12
        y2 += 1./12
    
    #Calculate position angle
    theta = np.sign(xy) * 0.5*abs( np.arctan2(2*xy, x2-y2) ) + np.pi/2
    
    #Calculate the semimajor & minor axes
    c1 = 0.5*(x2+y2)
    c2 = np.sqrt( ((x2-y2)/2)**2 + xy**2 )
    arms = np.sqrt( c1 + c2 )
    brms = np.sqrt( c1 - c2 )

    if not rms:
        dmax = np.sqrt( np.max( ((xs-x0)**2+(ys-y0)**2) ) )
        dmax = np.max((dmax, 1)) #Account for 1-pixel detections
        bmax = (brms/arms)*dmax   
        return geometry.ellipse(x0=x0,y0=y0,a=dmax,b=bmax,theta=theta)

    return geometry.ellipse(x0=x0,y0=y0,a=arms,b=brms,theta=theta)


class Source():
    def __init__(self, label, cslice):
        self.ellipse_max = None
        self.ellipse_rms = None
        self.ellipse_rms_weighted = None
        self.xs = None
        self.ys = None
        self.Is = None
        self.cslice = cslice
        self.label = label
    def get_crds(self,clusters, mask=None):
        if ((self.xs is None)*(self.ys is None)):
            xs, ys = np.meshgrid(np.arange(self.cslice[1].start,self.cslice[1].stop),
                                 np.arange(self.cslice[0].start,self.cslice[0].stop))
            cond = clusters[self.cslice]==self.label
            
            #Mask condition
            if mask is not None:
                cond *= mask[self.cslice] == 0
            
            self.xs = xs[cond]
            self.ys = ys[cond]
        return self.xs, self.ys
    def get_data(self, data, clusters, mask=None):
        xs, ys = self.get_crds(clusters, mask=mask)
        if self.Is is None:
            self.Is = data[ys, xs]
        return self.Is
    def get_ellipse_max(self, clusters, mask=None):
        xs, ys = self.get_crds(clusters, mask=mask)
        self.ellipse_max=fit_ellipse(xs,ys,weights=None,rms=False)
        return self.ellipse_max
    def get_ellipse_rms(self, mask=None):
        xs, ys = self.get_crds(clusters, mask=mask)
        self.ellipse_rms=fit_ellipse(self.xs,self.ys,weights=None,rms=True)
        return self.ellipse_rms
    def get_ellipse_rms_weighted(self, clusters, data, mask=None):
        Is = self.get_data(data, clusters, mask=mask) #xs & ys are set in get_data
        self.ellipse_max_weighted=fit_ellipse(self.xs,self.ys,weights=Is,rms=True)
        return self.ellipse_max_weighted


#==============================================================================


def perform_convolution(xmin, xmax, ymin, ymax, R, kernel, dshape):
    
    #Expand box
    xmax2 = xmax + R
    xmin2 = xmin - R
    ymin2 = ymin - R
    ymax2 = ymax + R
    
    #Look for boundary overlap
    xoverlap1 = np.max((0, -xmin2))           #Negative x boundary overlap
    xoverlap2 = np.max((0, xmax2-dshape[1]))  #Positive x boundary overlap
    yoverlap1 = np.max((0, -ymin2))           #Negative y boundary overlap
    yoverlap2 = np.max((0, ymax2-dshape[0]))  #Positive y boundary overlap
    
    #Crop
    xmax2 = int(np.min((xmax2, dshape[1])))
    ymax2 = int(np.min((ymax2, dshape[0])))
    xmin2 = int(np.max((xmin2, 0)))
    ymin2 = int(np.max((ymin2, 0)))
      
    cnv = fftconvolve(np.array(threshed[ymin2:ymax2,xmin2:xmax2]),
                                        kernel, mode='same').astype('int')
        
    conv[ymin:ymax, xmin:xmax] = cnv[R-yoverlap1:cnv.shape[0]-R+yoverlap2,
                                            R-xoverlap1:cnv.shape[1]-R+xoverlap2]
            
def perform_convolution2(xmin, xmax, ymin, ymax, R, kernel, dshape):
    
    #Expand box
    xmax2 = xmax + R
    xmin2 = xmin - R
    ymin2 = ymin - R
    ymax2 = ymax + R
    
    #Look for boundary overlap
    xoverlap1 = np.max((0, -xmin2))           #Negative x boundary overlap
    xoverlap2 = np.max((0, xmax2-dshape[1]))  #Positive x boundary overlap
    yoverlap1 = np.max((0, -ymin2))           #Negative y boundary overlap
    yoverlap2 = np.max((0, ymax2-dshape[0]))  #Positive y boundary overlap
    
    #Crop
    xmax2 = int(np.min((xmax2, dshape[1])))
    ymax2 = int(np.min((ymax2, dshape[0])))
    xmin2 = int(np.max((xmin2, 0)))
    ymin2 = int(np.max((ymin2, 0)))
      
    #Convolve and fill
    cnv = fftconvolve(np.array(conv[ymin2:ymax2,xmin2:xmax2]),
                                        kernel, mode='same').astype('int')
    conv2[ymin:ymax, xmin:xmax] = cnv[R-yoverlap1:cnv.shape[0]-R+yoverlap2,
                                            R-xoverlap1:cnv.shape[1]-R+xoverlap2]
    


def label_chunk(xmin, xmax, ymin, ymax, multiplier, dshape, overlap, mpts):
    
    t0 = time.time()
    
    #thresh = np.array(conv[ymin:ymax, xmin:xmax] > mpts, dtype='int')
    thresh = np.array(conv2[ymin:ymax, xmin:xmax])
    
    t1 = time.time() - t0
    
    labels = label(thresh)[0]

    t2 = time.time() - t1 - t0
    
    labeled[ymin:ymax, xmin:xmax-overlap] += (labels[:,:-overlap]).astype('complex')*multiplier        
    
    t3 = time.time() - t2 - t1 - t0
    
    print('labeling', t3, t2, t1)
    
    lock.acquire()
    labeled[ymin:ymax, xmax-overlap:xmax] += (labels[:,-overlap:
                                            ]).astype('complex')*multiplier
    lock.release()

                               
    
def stitch(labels_complex, mpts_threshed):
    
    t0 = time.time()
    
    sids = np.array([u for u in np.unique(labels_complex) if ((u.real!=0) * (u.imag!=0))])

    clusters = []
    #Create initial cluster groups
    for uid in sids:
        added = False
        for cluster in clusters:
            for entry in cluster:
                if ((uid.real == entry.real) or (uid.imag == entry.imag)):
                    cluster.append(uid)
                    added = True
                    break
            if added:
                break
        if not added:
            clusters.append([uid])
    
    t1 = time.time() - t0
    print('stitch: made first clusters in %i seconds' % t1)
    
    #Now merge the groups
    finished = False
    arr_ignore = np.zeros(len(clusters), dtype='bool') 
    while not finished:
        changes = False
        #Get indices of clusters we want to check
        indices = np.arange(arr_ignore.size)[~arr_ignore]
        clusters_ = [clusters[k] for k in indices]
        #Loop over unique cluster pairs that are not in arr_ignore
        for i, c1 in enumerate(clusters_):
            for j in range(i+1, indices.size):
                c2 = clusters_[j]
                for entry1 in c1[:len(c1)]:
                    if entry1.real in [ec2.real for ec2 in c2]:
                        clusters[indices[i]].extend([c for c in c2 if c not in c1])
                        arr_ignore[indices[j]] = True
                        changes = True
                        break
                    elif entry1.imag in [ec2.imag for ec2 in c2]:
                        clusters[indices[i]].extend([c for c in c2 if c not in c1])
                        arr_ignore[indices[j]] = True
                        changes = True
                        break
                    
        #Check for completion
        if changes == False:
            finished = True
            
    #Select merged clusters        
    clusters = [c for i, c in enumerate(clusters) if not arr_ignore[i]]
    
    t2 = time.time() - t1 - t0
    print('stitch: merged all clusters in %i seconds' % t2)
        
    slices_real = find_objects(labels_complex.real.astype('int'))
    slices_imag = find_objects(labels_complex.imag.astype('int'))
        
    slices_real_ov = [slices_real[i-1] for i in sids.real.astype('int')]
    slices_imag_ov = [slices_imag[i-1] for i in sids.imag.astype('int')]
    
    slices_real_grouped = []
    slices_imag_grouped = []
    for cluster in clusters:
        creals = [c.real for c in cluster]
        cimags = [c.imag for c in cluster]
        slices_real_grouped.append([s for i, s in enumerate(slices_real_ov) if
                                                    sids[i].real in creals])
        slices_imag_grouped.append([s for i, s in enumerate(slices_imag_ov) if 
                                                    sids[i].imag in cimags])
    
    t2b = time.time() - t1 - t0 - t2
    print('stitch: measured cluster boundaries in %i seconds' % t2b)
        
    #Relabel the clusters as unique integers
    labels = np.zeros_like(labels_complex, dtype='int')
    labels = np.array(labels_complex.imag, dtype='int')
    labels += np.array(labels_complex.real, dtype='int')
    
    for i, c in enumerate(clusters):
        for j, val in enumerate(c):
            cond1 = labels_complex[slices_real_grouped[i][j]].real ==  val.real
            cond2 = labels_complex[slices_imag_grouped[i][j]].imag ==  val.imag
            (labels[slices_real_grouped[i][j]])[cond1] = int(c[0].real)
            (labels[slices_imag_grouped[i][j]])[cond2] = int(c[0].real)
            
    t3 = time.time() - t2 -t1 -t0 - t2b
    print('stitch: relabeled clusters in %i seconds' % t3)
    
    #Retrieve the DBSCAN clusters (core points)
    clusters = labels * mpts_threshed
        
    #Order cluster labels
    uids = np.unique(clusters)[1:]    
    slices_ = find_objects(clusters)

    Nclusters = len(uids)
    slices = slices_[:Nclusters]
    tofill = [i for i, s in enumerate(slices) if s is None]
    tomove = [i for i, s in enumerate(slices_) if ((s is not None)*(i>=Nclusters))]
    
    for i, ind_fill in enumerate(tofill):
        slices[ind_fill] = slices_[tomove[i]]
        cond = clusters[slices[ind_fill]] == tomove[i]+1
        clusters[slices[ind_fill]][cond] = ind_fill+1
    
    return clusters, slices, labels
                    
                    

def init(l,thresh_memmap,conv_memmap,label_memmap, conv_memmap2):
    global lock, threshed, conv, labeled, conv2
    lock = l
    threshed = thresh_memmap
    conv = conv_memmap
    conv2 = conv_memmap2
    labeled = label_memmap
def dbscan_conv(data, thresh, eps, mpts, Nthreads=1,
                                         meshsize=None,
                                         thresh_type='absolute',
                                         rms=None,
                                         minCsize=5,
                                         dist_labeling=False,
                                         memmap_thresh=False):
    
    if meshsize is None:
        meshsize = BUFFSIZE
    
    #Do some argument checking
    if Nthreads < 1:
        raise ValueError('Nthreads<1.')
    '''
    if Nthreads == 1:
        try:
            assert(meshsize is None)
        except AssertionError:
            raise ValueError('meshsize should be None for Nthreads=1.')
    '''
    if Nthreads > 1:
        try:
            assert(type(meshsize)==int) 
        except AssertionError:
            raise ValueError('meshsize must be an integer.')
        try:
            assert(meshsize>0)
        except AssertionError:
            raise ValueError('meshsize must be >0.')
    
    try:
        assert( (type(eps)==int) or (type(eps)==float) )
    except:
        raise TypeError('eps parameter should either be an integer or double.')
    
    try:
        assert(type(mpts)==int)
    except:
        raise TypeError('mpts parameter should either be an integer or double.')
    
    if thresh_type == 'absolute':
        try:
            assert(rms is None)
            rms = 1
        except:
            TypeError('rms map should be None if thresh_type=absolute.')
    elif thresh_type == 'SNR':
        try:
            assert(rms is not None)
        except:
            TypeError('rms map should not be None if thresh_type=SNR.')
    else:
        raise ValueError("Allowable thresh_type(s): 'absolute', 'SNR'.")
        
                   
    #Create a convolution kernel
    kernel_conv = Tophat2DKernel(eps).array
    kernel_conv[kernel_conv!=0] = 1  #Require unit height
    
    #Create a tempory file for the memory map
    temppath = tempfile.mkdtemp()

    pool = None                                               
    try:
               
        #print('Creating memmaps...')
        
        if memmap_thresh:
            #Create a memory map for the thresholded image
            threshfilename = os.path.join(temppath, 'temp1.memmap')
            thresh_memmap = np.memmap(threshfilename, dtype='int', mode='w+',
                                    shape=data.shape)
            thresh_memmap[:] = (data > thresh*rms).astype('int')
        else:
            thresh_memmap = (data > thresh*rms).astype('int')
        
        #Create a memory map for the first convolved image
        convfilename = os.path.join(temppath, 'temp2.memmap')
        conv_memmap = np.memmap(convfilename, dtype='int', mode='w+',
                                shape=data.shape)
        
        #Create a memory map for the second convolved image
        convfilename2 = os.path.join(temppath, 'temp3.memmap')
        conv_memmap2= np.memmap(convfilename2, dtype='int', mode='w+',
                                shape=data.shape)

        
        if dist_labeling:
            #Create a memory map for the labeled image
            labelfilename = os.path.join(temppath, 'temp4.memmap')
            label_memmap = np.memmap(labelfilename, dtype='complex', mode='w+',
                                    shape=data.shape)
            label_memmap[:] = np.zeros_like(data, dtype='complex')
        else:
            label_memmap = None
        
        
        #Make a process pool
        l = multiprocessing.Lock()
        pool = multiprocessing.Pool(processes=Nthreads,initializer=init,
                                    initargs=(l,thresh_memmap, conv_memmap, 
                                              label_memmap, conv_memmap2))

        
        #Create the chunk boundary arrays
        xmins = np.arange(0, data.shape[1], meshsize)
        xmaxs = np.arange(meshsize, data.shape[1]+meshsize, meshsize) 
        ymins = np.arange(0, data.shape[0], meshsize) 
        ymaxs = np.arange(meshsize, data.shape[0]+meshsize, meshsize)
        bounds_list = [[xmins[i],xmaxs[i],ymins[j],ymaxs[j]]
                        for i in range(xmins.size) for j in range(ymins.size)]

        
        #print('Convolving...')
                
        #Create a function to perform the convoluton
        pfunc = partial(perform_convolution, kernel=kernel_conv,
                         dshape=data.shape, R=2*int(np.ceil(eps)))                       
        pool.starmap(pfunc, bounds_list)
         
        #Apply minpts condition to the convolved map
        cond1 = conv_memmap < mpts+1 
        conv_memmap[cond1] = 0
        conv_memmap[~cond1] = 1
                
        
        #Re-apply the convolution
        pfunc = partial(perform_convolution2, kernel=kernel_conv,
                         dshape=data.shape, R=2*int(np.ceil(eps)))  
        pool.starmap(pfunc, bounds_list)
                
        #Get cluster regions
        conv_memmap2 = (conv_memmap2 >= 1).astype('int')#minCsize
 
        
        #print('Labeling...')
        
        
        if dist_labeling:
            
            raise ValueError('dis_labeling is currently inoperational')
                    
            #Get strip coordinates
            overlap = 4
            step = (data.shape[1]/Nthreads)
            xmins = np.linspace(0, data.shape[1]-step, Nthreads, dtype='int') 
            xmaxs = [xmin+overlap for xmin in xmins[1:]]; xmaxs.append(data.shape[1])
            ymins = np.zeros_like(xmins)
            ymaxs = np.ones_like(xmaxs)*data.shape[0]
            bounds_list = [[xmins[i],xmaxs[i],ymins[i],ymaxs[i]]
                            for i in range(xmins.size)]
            for i, b in enumerate(bounds_list):
                if (i+1)%2!=0:
                    b.extend([1])
                else:
                    b.extend([1j])
            
            #Create a function to perform the labeling
            pfunc = partial(label_chunk, dshape=data.shape, overlap=overlap,
                                                 mpts=mpts)
            pool.starmap(pfunc, bounds_list)
            
            #Convert to unique identifiers
            total = 0
            for i, b in enumerate(bounds_list):
                cutout = label_memmap[b[2]:b[3], b[0]:b[1]]
                if (i+1)%2!=0:
                    cutout[cutout.real!=0] += total
                    total += np.max(cutout.real)
                else:
                    cutout[cutout.imag!=0] += total*1j
                    total += np.max(cutout.imag)
    
            #print('Stitching...')
            
            #Perform stitching & get slice        
            clusters, slices, labels = stitch(label_memmap, thresh_memmap*conv_memmap)
            
        else:
            
            #Do the labeling on one processor
            labeled, Nlabels = label(conv_memmap2)
            slices_labeled = find_objects(labeled)
            
            #Select the clusters as core points
            corepoints = conv_memmap * thresh_memmap
            clusters = corepoints * labeled
            
            #Find unique cluster identifiers
            uids_ = np.unique(clusters)
            if uids_[0]==0:  #Remove 0 for cluster IDs
                uids_ = uids_[1:]
                
            #print('%i Unique clusters (1)' % uids_.size)
                    
            #Remove clusters that are too small
            slices_clus = [s for s in find_objects(clusters) if s is not None]
            clens = [np.sum(clusters[s]==cid) for cid, s in zip(uids_, slices_clus)]
            
            #print(len([c for c in clens if c==0]))
            
            uids = np.array([u for i, u in enumerate(uids_) if clens[i]>=minCsize])  
            #Delete the clusters than were eliminated
            if uids.size != uids_.size:          
                for i, s in enumerate(slices_clus):
                    if (i+1) not in uids:
                        clusters[s][clusters[s]==(i+1)]=0
            Nclusters = uids.size
            
            #print('%i Unique clusters (2)' % uids.size)
                                
            #Check if there are any non-continuities in the labeling
            if (uids[-1] != Nlabels) or (Nlabels!=uids.size):
                
                #print('hello1')
                
                #Select label_ids to keep
                label_ids = np.arange(1,Nlabels+1)
                label_slices_keep = np.ones(Nlabels, dtype='bool')
                for i, s in enumerate(slices_labeled):
                    if (i+1) not in uids:
                        
                        #Delete the labels with no clusters
                        labeled[s][labeled[s]==(i+1)]=0
                        label_slices_keep[i] = False
                        
                #Do the relabling
                ids_to_fill = (label_ids[:Nclusters])[~label_slices_keep[:Nclusters]]
                ids_to_replace = label_ids[Nclusters:][label_slices_keep[Nclusters:]]

                
                #print(ids_to_replace.size,len(ids_to_fill))
                #assert(ids_to_replace.size == len(ids_to_fill))
                
                for i, id_fill in enumerate(ids_to_fill):
                    #Get slice of object to relable
                    slice_ = slices_labeled[ids_to_replace[i]-1]
                    #Do the relable
                    cond = labeled[slice_] == ids_to_replace[i]
                    labeled[slice_][cond] = id_fill
                    #Update the slices
                    slices_labeled[id_fill-1] = slices_labeled[ids_to_replace[i]-1]
                
                #Finish up
                clusters = corepoints * labeled
                slices = slices_labeled[:Nclusters]
                
            else:
                #print('hello2')
                slices = slices_labeled
            
            
        #Create source instances
        sources = []
        for i, slice_ in enumerate(slices):
            sources.append( Source( i+1, slice_) )
        
        
        #print('Finishing')
        #Remove the memmaps from memory
        del thresh_memmap
        del conv_memmap
        del conv_memmap2
        del label_memmap
            
    finally:
        
        #Remove the temporary directory
        shutil.rmtree(temppath)
        
        #Shut down the pool
        if pool is not None:
            pool.close()
            pool.join()
        
    return clusters, labeled, sources


#==============================================================================


#Test
if __name__ == '__main__':
    
    meshsize = 3000
    Nthreads = 4
    thresh = 1

    from deepscan import utils, SB
    from ppy.dbscan import required_minpts
    import matplotlib.pyplot as plt
    SBn = 26.9
    ps=0.187
    mzero=30
    kappa = 25
    eps = 10
    data = utils.read_fits('/Users/danjampro/Dropbox/phd/data/VLSB_Ga.fits')
    #data = utils.read_fits('/Users/danjampro/Dropbox/phd/NGVS/data/NGVS+0+0.G.fits')#[2000:12000,2000:12000]
    
    mpts = required_minpts(kappa=kappa, eps=eps/ps, tmin=1, tmax=np.inf, sign=1)
    eps = eps/ps
    rms = np.ones_like(data) * SB.SB2Counts(SBn,ps,mzero)
    
    
    t0 = time.time()
    clusters, labeled, sources = dbscan_conv(data, thresh=thresh, eps=eps, mpts=mpts, meshsize=meshsize,
                       Nthreads=Nthreads, rms=rms)
    t1 = time.time() - t0
    print('DBSCAN time: %.0f seconds' % t1)
    print(np.max(clusters), len(np.unique(clusters)))
    
    
    ellipses = [source.get_ellipse_max(clusters) for source in sources]
    
    plt.figure()
    clusters_ = np.copy(clusters).astype('float')
    clusters_[clusters_==0] = float(np.nan)
    #plt.imshow(clusters_)
    [e.draw(color='r') for e in ellipses]
    


   
    
    
    
        
