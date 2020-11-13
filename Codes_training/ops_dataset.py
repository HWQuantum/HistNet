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
import cv2
import math
import time
import scipy.stats
import argparse
from datetime import datetime

def create_hist(patch_depth_LR_norm , patch_intensity_norm, ppp):
    # --- Inputs ---
    # patch_depth_LR_norm : dictionary of depth image of size [Nx, Ny]
    # patch_intensity_norm : dictionary of corresponding intensity image normalized from 0 to 1
    # ppp : total nb of photon counts per pixel (average over the image)

    ## --- Outputs ---
    #  hist_patch_depth : dictionary of histograms of size [Nx, Ny, Nbins]

    
    sigma = 0.5714 #standard deviation 
    Nbins = 15
    test_depth = patch_depth_LR_norm[0]
    Nx = test_depth.shape[0]
    Ny = test_depth.shape[1]
    
    hist_patch_depth = {}
    
    # Definition of "array_bin": array taking all possible value that can appear in the histograms, 
    # with respect to a certain precision 
    precision = 100 #precision per bin 
    array_x = -16 + np.linspace(0, 32*precision, num = 32*precision)/precision
    array_exp = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(- np.square(array_x)/np.square(sigma))
    array_bin = np.zeros(len(array_exp))
    for i in range(len(array_exp)-precision+1):
        array_bin[i] = np.sum(array_exp[range(i , i+precision , 1)])


    # Attribution of the values by picking in array_bin
    nb_patches = len(patch_depth_LR_norm)
    for index_patch in range(nb_patches):
        I_up            = patch_depth_LR_norm[index_patch]
        intensity_image = patch_intensity_norm[index_patch]
        average_intensity = np.mean(intensity_image)
        ppp_array = []
        bins = np.zeros((Nx,Ny,Nbins))
        for i in range(Nx):
            for j in range(Ny):
                depth = I_up[i,j]
                index_array = precision * (16 - Nbins*depth)
                for index_bin in range(Nbins):
                    index_array_bin = index_array + precision*index_bin
                    bins[i,j,index_bin] = array_bin[np.int(index_array_bin)]

                bins[i,j,:] = bins[i,j,:] * intensity_image[i,j] * ppp / average_intensity / np.sum(np.squeeze(bins[i,j,:]))
                #bins[i,j,:] = bins[i,j,:] * intensity_image[i,j] * ppp / np.sum(np.squeeze(bins[i,j,:]))
                ppp_array.append(np.sum(bins[i,j,:]))
        ppp_mean = np.mean(ppp_array)
        

        hist_patch_depth[index_patch] = bins


    return hist_patch_depth


def create_noise(Histogram_training_depth_LR, SBR, ambient_type):
    # --- Inputs ---
    # Histogram_training_depth_LR : dictionary of histograms of size [Nx, Ny, Nbins]
    # SBR : number of photons in the peak / background below the peak. We simulate a peak that spread over approx. 3 bins.
    # ambient_type : no_ambient, constant_b (same background level for all pixels), constant_SBR (same SBR for all pixels)

    ## --- Outputs ---
    #  Histogram_training_depth_LR_noisy : dictionary of noisy histograms of size [Nx, Ny, Nbins]

    batch_idx = len(Histogram_training_depth_LR)

    Histogram_training_depth_LR_noisy = {}
    b_val_array = []
    for index in range(batch_idx):
        histogram = Histogram_training_depth_LR[index] 
        Nx = histogram.shape[0]
        Ny = histogram.shape[1]
        Nbins = histogram.shape[2]

        # Define ambient signal
        b = np.zeros((Nx, Ny, Nbins))
        if ambient_type == 'no_ambient':
            b = np.zeros((Nx, Ny,Nbins))

        elif ambient_type == 'constant_SBR':
            for i in range(Nx):
                for j in range(Ny):
                    #b_val = np.sum(np.squeeze(histogram[i,j,:])) / (Nbins*SBR)
                    b_val = np.sum(np.squeeze(histogram[i,j,:]))/(3*SBR)
                    b_val_array.append(b_val)
                    b[i,j,:] = np.ones(Nbins)*b_val
        b_val_mean = np.mean(b_val_array)
         
            
        # Define noisy histogram 
        histogram_ambient = np.zeros((Nx, Ny, Nbins))
        for i in range(Nx):
            for j in range(Ny):
                histogram_ambient[i,j,:] = histogram[i,j,:] + b[i,j,:]
        histogram_noisy = np.random.poisson(histogram_ambient)

        Histogram_training_depth_LR_noisy[index] = histogram_noisy

    return Histogram_training_depth_LR_noisy

def center_of_mass(Histogram_training_depth_LR, type):

    if type =='dict':
        nb_images = len(Histogram_training_depth_LR)
        depth_LR = Histogram_training_depth_LR[0]
        Nx_LR = int(depth_LR.shape[0])
        Ny_LR = int(depth_LR.shape[1])
        Nbins = depth_LR.shape[2]
        Depth_images = {}

    elif type =='one_image':
        depth_LR = Histogram_training_depth_LR
        nb_images = 1
        Nx_LR = int(depth_LR.shape[0])
        Ny_LR = int(depth_LR.shape[1])
        Nbins = depth_LR.shape[2]
        Depth_images = np.zeros((nb_images, Nx_LR, Ny_LR))

    for index in range(nb_images):
        if type =='dict':
            depth_LR = np.float32(Histogram_training_depth_LR[index])
        elif type =='one_image':
            depth_LR = np.float32(Histogram_training_depth_LR)
        depth_image = np.zeros((Nx_LR,Ny_LR))
        denominator = np.zeros((Nx_LR,Ny_LR))
        numerator = np.zeros((Nx_LR,Ny_LR))

        for i in range(Nx_LR):
            for j in range(Ny_LR):
                # Define maximum symmetric window (range_center_of_mass) around maximum (pos_max)
                pos_max = np.argmax(np.squeeze(depth_LR[i,j,:]))
                index_bin = 0
                while pos_max + index_bin < Nbins and pos_max - index_bin > 0 and index_bin < 2:  
                    index_bin = index_bin + 1

                if index_bin==0:
                    depth_image[i,j]= pos_max
                else:
                    range_center_of_mass = range(pos_max-index_bin , pos_max + index_bin, 1) 
                    #range_center_of_mass = range(pos_max - 2 , pos_max + 2, 1) 
                    
                    # Define b 
                    b = np.median(np.squeeze(depth_LR[i,j,:]))
                    for t in range_center_of_mass:     
                        numerator[i,j] = numerator[i,j] + t * np.maximum(depth_LR[i,j,t] - b, 0)
                        denominator[i,j] = denominator[i,j] + np.maximum(depth_LR[i,j,t] - b, 0)
                    if denominator[i,j] == 0:
                        depth_image[i,j] = 0
                    else:
                        depth_image[i,j] = numerator[i,j] / denominator[i,j]

                    
        depth_image = np.float32(depth_image)
        depth_image = depth_image / 15

        if type =='dict':
            Depth_images[index] = depth_image
        elif type == 'one_image':
            Depth_images[index, :,:] = depth_image
    return Depth_images


def create_patches(intensity_training_aug , depth_training_aug, image_size, stride):
    patch_training_intensity = {}
    patch_training_depth = {}
    patch_idx = 0
    for index in range(0 , len(intensity_training_aug) , 1):
        intensity_im    = intensity_training_aug[index]
        depth_im        = depth_training_aug[index]
        Nx = intensity_im.shape[0]
        Ny = intensity_im.shape[1]

        if Nx != depth_im.shape[0] or Ny != depth_im.shape[1]:
            raise Exception('Corresponding intensity and depth of dif size ..')

        for y in range(0 , Ny - image_size , stride):
            for x in range(0 , Nx - image_size , stride):  
                patch_intensity = intensity_im  [x:x+image_size , y:y+image_size]
                patch_depth     = depth_im      [x:x+image_size , y:y+image_size]
                
                patch_training_intensity[patch_idx] = patch_intensity
                patch_training_depth    [patch_idx] = patch_depth    
                patch_idx = patch_idx + 1
    #print('#Training_Patches = '+str(patch_idx))
    return patch_training_intensity, patch_training_depth

def second_peak_histogram(histogram_initial, type, level):
    #level = 12 for hammer, 4 for training, 0.3 for 2depth)
    if type == 'dict':
        Nimage = len(histogram_initial)
        second_histogram_dict = {}
        first_histogram_dict = {}
    elif type == 'one_image':
        Nimage = 1
    
    for index in range(Nimage):
        if type == 'dict':
            histogram = histogram_initial[index]
        elif type == 'one_image':
            histogram = histogram_initial
        Nx_LR = int(histogram.shape[0])
        Ny_LR = int(histogram.shape[1])
        Nbins = histogram.shape[2]
        first_histogram = np.zeros((Nx_LR,Ny_LR,Nbins))
        second_histogram = np.zeros((Nx_LR,Ny_LR,Nbins))
        new_histogram = np.zeros((Nx_LR,Ny_LR,Nbins))

        for i in range(Nx_LR):
                for j in range(Ny_LR):
                    pos_max = np.argmax(np.squeeze(histogram[i,j,:]))
                    range_center_of_mass = range(max(pos_max-2, 0), min(pos_max+3,Nbins))
                    b = np.median(np.squeeze(histogram[i,j,:]))
                    new_histogram[i,j,:] = np.squeeze(histogram[i,j,:])
                    new_histogram[i,j,range_center_of_mass] = b

                    # First Histogram condition on size peak to reduce noise. 
                    # if max(histogram[i,j,:]) < b + level/2*np.sqrt(b): 
                    #     first_histogram[i,j,:] = np.zeros(Nbins) 
                    # else :
                    #     first_histogram[i,j,:] = histogram[i,j,:]
                    first_histogram[i,j,:] = histogram[i,j,:]
                    
                    # Second Histogram condition on size peak.
                    if max(new_histogram[i,j,:]) < b + level * np.sqrt(b): 
                        second_histogram[i,j,:] = np.zeros(Nbins) 
                    else :
                        second_histogram[i,j,:] = new_histogram[i,j,:]
        if type == 'dict':
            second_histogram_dict[index] = second_histogram
            first_histogram_dict[index] = first_histogram

    if type == 'one_image':    
        return first_histogram, second_histogram
    if type == 'dict':
        return first_histogram_dict, second_histogram_dict




# def create_hist(patch_depth_LR_norm , patch_intensity_norm, intensity_level):
#     # --- Inputs ---
#     # patch_depth_LR_norm : dictionary of depth image of size [Nx, Ny]
#     # patch_intensity_norm : dictionary of corresponding intensity image normalized from 0 to 1
#     # intensity_level : nb of photon counts

#     ## --- Outputs ---
#     #  hist_patch_depth : dictionary of histograms of size [Nx, Ny, Nbins]


#     sigma = 0.5714 #standard deviation 
#     Nbins = 15
#     test_depth = patch_depth_LR_norm[0]
#     Nx = test_depth.shape[0]
#     Ny = test_depth.shape[1]
    
#     hist_patch_depth = {}
    
#     # Definition of "array_bin": array taking all possible value that can appear in the histograms, 
#     # with respect to a certain precision 
#     precision = 100 #precision per bin 
#     array_x = -(Nbins+1) + np.linspace(0, 32*precision, num = (Nbins+1)*2*precision)/precision
#     array_exp = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(- np.square(array_x)/np.square(sigma))
#     array_bin = np.zeros(len(array_exp))
#     for i in range(len(array_exp)-precision+1):
#         array_bin[i] = np.sum(array_exp[range(i , i+precision , 1)])


#     # Attribution of the values by picking in array_bin
#     nb_patches = len(patch_depth_LR_norm)
#     for index_patch in range(nb_patches):
#         I_up            = patch_depth_LR_norm[index_patch]
#         intensity_image = patch_intensity_norm[index_patch]
#         mean_intensity = np.mean(np.squeeze(intensity_image))
#         bins = np.zeros((Nx,Ny,Nbins))
#         for i in range(Nx):
#             for j in range(Ny):
#                 depth = I_up[i,j]
#                 index_array = precision * (Nbins + 1 - Nbins*depth)
#                 for index_bin in range(Nbins):
#                     index_array_bin = index_array + precision*index_bin
#                     bins[i,j,index_bin] = array_bin[np.int(index_array_bin)]
#                 #if type_creation == 'lindell':   
#                 #    bins[i,j,:] = bins[i,j,:] * intensity_image[i,j] * intensity_level / (mean_intensity*np.sum(np.squeeze(bins[i,j,:])))
#                 #else:
#                 bins[i,j,:] = bins[i,j,:] * intensity_image[i,j] * intensity_level / np.sum(np.squeeze(bins[i,j,:]))

#         hist_patch_depth[index_patch] = bins


#     return hist_patch_depth


# def create_noise(Histogram_training_depth_LR,SBR_mean, ambient_type):
#     # --- Inputs ---
#     # Histogram_training_depth_LR : dictionary of histograms of size [Nx, Ny, Nbins]
#     # SBR_mean : mean SBR of an image. We approximate that there is the same ambient signal in every pixel of a patch. 
#     # no_ambient : 0 to add an ambient signal, 0 otherwise

#     ## --- Outputs ---
#     #  Histogram_training_depth_LR_noisy : dictionary of noisy histograms of size [Nx, Ny, Nbins]

#     batch_idx = len(Histogram_training_depth_LR)

#     Histogram_training_depth_LR_noisy = {}
#     for index in range(batch_idx):
#         histogram = Histogram_training_depth_LR[index] 
#         #intensity_image = patch_intensity_norm[index_patch]
#         Nx = histogram.shape[0]
#         Ny = histogram.shape[1]
#         Nbins = histogram.shape[2]

#         # Define ambient signal
#         b = np.zeros((Nx, Ny, Nbins))
#         if ambient_type == 'no_ambient':
#             b = np.zeros((Nx, Ny,Nbins))

#         elif ambient_type == 'constant_b': # case Lindell?
#             #b_val = np.sum(np.squeeze(histogram[:,:,:])) / (Nbins*SBR_mean*Nx*Ny)
#             b_val = SBR_mean
#             if index == 0 :
#                 SBR = np.sum(np.squeeze(histogram[:,:,:])) / (Nbins*b_val*Nx*Ny)
#                 print(SBR)
#             b = b_val*np.ones((Nx, Ny, Nbins))

#         elif ambient_type == 'constant_SBR': # case Hammer data
#             for i in range(Nx):
#                 for j in range(Ny):
#                     b_val = np.sum(np.squeeze(histogram[i,j,:])) / (Nbins*SBR_mean)
#                     b[i,j,:] = np.ones(Nbins)*b_val
#                     #print(b_val)

            
#         # Define noisy histogram 
#         histogram_ambient = np.zeros((Nx, Ny, Nbins))
#         for i in range(Nx):
#             for j in range(Ny):
#                 histogram_ambient[i,j,:] = histogram[i,j,:] + b[i,j,:]
#         histogram_noisy = np.random.poisson(histogram_ambient)

#         Histogram_training_depth_LR_noisy[index] = histogram_noisy

#     return Histogram_training_depth_LR_noisy