import argparse
from datetime import datetime
import os
import yaml
import scipy
from scipy import io as sio
import scipy.misc
import numpy as np
import glob 
import random
import matplotlib.pyplot as plt
import skimage
import skimage.transform
import math
import time
from ops_dataset import *

import time

def create_histograms(config,Dir_import, intensity_level, SBR_mean):
    
    # --- Outputs ---    
    # Histogram_training_depth_LR_DS    : LR Histograms for Training 
    # Histogram_validation_depth_LR_DS  : LR Histograms for Validation 
    # patch_training_intensity_norm     : Corresponding HR intensity for Training
    # patch_validation_intensity_norm   : Corresponding HR intensity for Validation
    # patch_training_depth_norm         : HR Histograms for Training Labels
    # patch_validation_depth_norm       : HR Histograms for Validation Labels

    image_size = config['image_size']
    stride = config['stride']
    scale = config['scale']
    

    # ---- 1. Import data --------------------------------------------------------------------------
    
    depth = sio.loadmat(os.path.join(Dir_import, 'depth_total.mat'))['depth_data_MPI']
    intensity = sio.loadmat(os.path.join(Dir_import,'intensity_total.mat'))['intensity_data_MPI']
    depth = np.squeeze(depth)
    intensity = np.squeeze(intensity)
    depth = depth#[1:3]
    intensity = intensity#[1:3]
    
    
    # ---- 2. Check NaN Inf values  + Crop  modulo scale -------------------------------------------------------------
    print('Check NaN Inf values...')
    count_nan , count_inf = 0 , 0
    depth_new = {}
    intensity_new = {}
    new_index = 0

    for index in range(0,depth.shape[0],1):
        depth_im = depth[index]
        intensity_im = intensity[index]
        
        if np.any(np.isnan(np.ndarray.flatten(depth_im))):
            count_nan = count_nan + 1
        elif np.any(np.isinf(np.ndarray.flatten(depth_im))):
            count_inf = count_inf + 1
        else:  
            # --- Crop images modulo 16
            h, w = depth_im.shape
            h = h - np.mod(h, 16)
            w = w - np.mod(w, 16)
            depth_im        = depth_im      [0:h, 0:w]
            intensity_im    = intensity_im  [0:h, 0:w]

            depth_new[new_index]        = depth_im#[0:192]
            intensity_new[new_index]    = intensity_im#[0:192]
            new_index = new_index + 1

    depth = depth_new
    intensity = intensity_new

    # ---- 2. Split Dataset into Training and Validation  ------------------------------------------
    print('Split Dataset into Training and Validation ratio...')
    ratio_train_test = 1/8
    random.seed(2000)
    indexes             =  np.random.permutation(len(depth))
    index_validation    = indexes[range(0 , int(ratio_train_test*len(depth)) , 1)]
    index_training      = indexes[range(int(ratio_train_test*len(depth)) , len(depth) , 1)]
    print(index_validation)
    intensity_validation = {}
    depth_validation     = {}
    new_index = 0 
    for index in index_validation:
        intensity_validation[new_index] = intensity[index]
        depth_validation[new_index]     = depth[index]
        new_index = new_index + 1

    intensity_training = {}
    depth_training     = {}
    new_index = 0 
    for index in index_training:
        intensity_training[new_index] = intensity[index]
        depth_training[new_index]     = depth[index]
        new_index = new_index + 1   

    print('Training : '+str(len(depth_training))+ ' images')
    print('Validation : '+str(len(depth_validation))+ ' images\n')

    # ---- 3. Flipping and rotation of Training and Validation dataset ---------------------------------------------
    print('Flipping and rotation of Training dataset...')
    intensity_training_aug = {}
    depth_training_aug = {}
    i = 0
    for index in range(0 , len(intensity_training),1):
        intensity_im        = intensity_training[index]
        depth_im            = depth_training[index]

        intensity_im_flip   = np.flipud(intensity_im)
        depth_im_flip       = np.flipud(depth_im)

        for angle in range(4):
            intensity_im        = np.rot90(intensity_im)
            depth_im            = np.rot90(depth_im)
            intensity_im_flip   = np.rot90(intensity_im_flip)
            depth_im_flip       = np.rot90(depth_im_flip)

            intensity_training_aug[i]   = intensity_im
            intensity_training_aug[i+1] = intensity_im_flip
            depth_training_aug[i]       = depth_im
            depth_training_aug[i+1]     = depth_im_flip

            i = i + 2

    print('Flipping and rotation of Validation dataset...')
    intensity_validation_aug = {}
    depth_validation_aug = {}
    i = 0
    for index in range(0 , len(intensity_validation),1):
        intensity_im        = intensity_validation[index]
        depth_im            = depth_validation[index]

        intensity_im_flip   = np.flipud(intensity_im)
        depth_im_flip       = np.flipud(depth_im)

        for angle in range(4):
            intensity_im        = np.rot90(intensity_im)
            depth_im            = np.rot90(depth_im)
            intensity_im_flip   = np.rot90(intensity_im_flip)
            depth_im_flip       = np.rot90(depth_im_flip)

            intensity_validation_aug[i]   = intensity_im
            intensity_validation_aug[i+1] = intensity_im_flip
            depth_validation_aug[i]       = depth_im
            depth_validation_aug[i+1]     = depth_im_flip

            i = i + 2
    print('Training : '+str(len(depth_training_aug))+ ' images')
    print('Validation : '+str(len(depth_validation))+ ' images\n')

    # ---- 4. Create Patches -------------------------------------------------------------------------
    print('Create Patches ...')

    patch_training_intensity , patch_training_depth = create_patches(intensity_training_aug , depth_training_aug , image_size , stride)
    print('Training : '+str(len(patch_training_intensity)) + ' patches')

    patch_validation_intensity , patch_validation_depth = create_patches(intensity_validation_aug , depth_validation_aug , image_size , stride)
    print('Validation : '+str(len(patch_validation_depth)) + ' patches\n')

    # ---- 5. Normalization ----------------------------------------------------------------------------
    print('Normalization ...')

    patch_training_depth_norm = {}
    patch_training_intensity_norm = {}
    count = 0
    index_save = 0
    for index in range(len(patch_training_intensity)):
        intensity = patch_training_intensity[index]
        depth = patch_training_depth[index]
        min_i = np.amin(intensity)
        max_i = np.amax(intensity)
        min_d = np.amin(depth)
        max_d = np.amax(depth)
        if min_i == max_i:
            count = count + 1
        elif min_d == max_d:
            count = count + 1
        else:
            patch_training_depth_norm[index_save] = (depth - min_d) / (max_d - min_d)
            patch_training_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
            index_save = index_save + 1


    index_save = 0
    patch_validation_depth_norm = {}
    patch_validation_intensity_norm = {}
    for index in range(len(patch_validation_intensity)):
        intensity = patch_validation_intensity[index]
        depth = patch_validation_depth[index]
        min_i = np.amin(intensity)
        max_i = np.amax(intensity)
        min_d = np.amin(depth)
        max_d = np.amax(depth)
        if min_i == max_i:
            count = count + 1
        elif min_d == max_d:
            count = count + 1
        else:
            patch_validation_depth_norm[index_save] = (depth - min_d) / (max_d - min_d)
            patch_validation_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
            index_save = index_save + 1


    print('Training : '+ str(len(patch_training_depth_norm))+' patches')
    print('Validation : '+ str(len(patch_validation_depth_norm))+' patches\n')

    # ---- 6. Create Histograms ------------------------------------------------------------------------
    Nbins = 15
    
    print("Create Histograms ...")
    Histogram_training_depth_LR = create_hist(patch_training_depth_norm , patch_training_intensity_norm,intensity_level)
    print('Training : ' + str(len(Histogram_training_depth_LR))+' histograms of size '+ str(Histogram_training_depth_LR[0].shape))


    Histogram_validation_depth_LR = create_hist(patch_validation_depth_norm , patch_validation_intensity_norm,intensity_level)
    print('Validation : '+ str(len(Histogram_validation_depth_LR))+' histograms of size '+ str(Histogram_training_depth_LR[0].shape)+'\n')

    # ---- 7. Add Noise ------------------------------------------------------------------------
    print("Create Noisy Histograms ...")
    ambient_type = 'constant_SBR'
    Histogram_training_depth_LR_noisy = create_noise(Histogram_training_depth_LR, SBR_mean, ambient_type)
    print('Training : ' + str(len(Histogram_training_depth_LR_noisy))+' histograms of size '+ str(Histogram_training_depth_LR_noisy[0].shape))

    Histogram_validation_depth_LR_noisy = create_noise(Histogram_validation_depth_LR, SBR_mean, ambient_type)
    print('Validation : '+ str(len(Histogram_validation_depth_LR_noisy))+' histograms of size '+ str(Histogram_validation_depth_LR_noisy[0].shape)+'\n')

    # ---- 8. Create HR intensity -------------------------------------
    print("Create Intensity ...")
    Histogram_validation_depth_LR_noisy
    nb_patches = len(Histogram_training_depth_LR_noisy)
    Intensity_training = {}
    for index in range(nb_patches):
        patch = Histogram_training_depth_LR_noisy[index]
        #print(patch.shape)
        Intensity_training[index] = np.sum(patch, 2)

    nb_patches = len(Histogram_validation_depth_LR_noisy)
    Intensity_validation = {}
    for index in range(nb_patches):
        patch = Histogram_validation_depth_LR_noisy[index]
        #print(patch.shape)
        Intensity_validation[index] = np.sum(patch, 2)
    print('Training : ' + str(len(Intensity_training))+' intensity maps of size '+ str(Intensity_training[0].shape))
    print('Validation : ' + str(len(Intensity_validation))+' intensity maps of size '+ str(Intensity_validation[0].shape)+'\n')

    # -- 9. Normalize intensity again -------------------------------------
    print('Normalize intensity ...')
    patch_training_intensity_norm = {}
    count = 0
    index_save = 0
    for index in range(len(Intensity_training)):
        intensity = Intensity_training[index]
        min_i = np.amin(intensity)
        max_i = np.amax(intensity)
        if min_i == max_i:
            count = count + 1
        else:
            patch_training_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
            index_save = index_save + 1

    index_save = 0
    patch_validation_intensity_norm = {}
    for index in range(len(Intensity_validation)):
        intensity = Intensity_validation[index]
        min_i = np.amin(intensity)
        max_i = np.amax(intensity)
        if min_i == max_i:
            count = count + 1
        else:
            patch_validation_intensity_norm[index_save] = (intensity - min_i) / (max_i - min_i)
            index_save = index_save + 1

    print('Training : '+ str(len(patch_training_intensity_norm))+' patches')
    print('Validation : '+ str(len(patch_validation_intensity_norm))+' patches\n')

    # ---- 10. Downsample Histograms -----------------------------------------------------------------------
    Histogram_training_depth_LR_DS = {}
    for patch_idx in range(0, len(Histogram_training_depth_LR_noisy)): 
        histogram = Histogram_training_depth_LR_noisy[patch_idx]
        Nx = histogram.shape[0]
        Ny = histogram.shape[1]
        Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale), Nbins))
        #print(Hist_LR.shape)
        i_x = 0
        for x in range(0,histogram.shape[0],scale):
            #print(i_x)
            i_y = 0
            for y in range(0,histogram.shape[1],scale):
                #print(i_y)
                for index_x in range(scale):
                    for index_y in range(scale):
                        Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale*scale) * histogram[x + index_x, y + index_y, :]
                i_y = i_y + 1
            i_x = i_x + 1

        Histogram_training_depth_LR_DS[patch_idx] = Hist_LR

    Histogram_validation_depth_LR_DS = {}
    for patch_idx in range(0, len(Histogram_validation_depth_LR_noisy)): 
        histogram = Histogram_validation_depth_LR_noisy[patch_idx]
        Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale),Nbins))
        i_x = 0
        for x in range(0,histogram.shape[0],scale):
            i_y = 0
            for y in range(0,histogram.shape[1],scale):
                for index_x in range(scale):
                    for index_y in range(scale):
                        Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale*scale) * histogram[x + index_x, y + index_y, :]
                i_y = i_y + 1
            i_x = i_x + 1

        Histogram_validation_depth_LR_DS[patch_idx] = Hist_LR
    
    return Histogram_training_depth_LR_DS, Histogram_validation_depth_LR_DS, patch_training_intensity_norm, patch_validation_intensity_norm, patch_training_depth_norm,patch_validation_depth_norm

