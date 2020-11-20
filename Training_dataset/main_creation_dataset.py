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
from create_histograms import *

# ---- To define --------------------------------------------------------------------------------
ppp = 4 ## ppp estimation
SBR_mean = 0.02 ## SBR estimation

# ---- Inputs --------------------------------------------------------------------------------
Directory = '/home/ar432/HistSR_Net_Repository/Training_Dataset/Dataset'
Dir_import = '/home/ar432/HistSR_Net_Repository/Training_Dataset/Raw_data_Middlebury_MPI'

image_size = 96
stride = 48
scale = 4
batch_size = 64

config = {}
config['image_size'] = image_size 
config['stride'] = stride
config['scale'] = scale 

Directory = os.path.join(Directory,'DATA_ppp_'+str(ppp)+'_SBR_'+str(SBR_mean)+'_test')
if not os.path.exists(Directory):
    os.mkdir(Directory)



# ---- 1. Create LR Histogram and HR Intensity ------------------------------------------------------
Histogram_training_depth_LR_DS, Histogram_validation_depth_LR_DS, patch_training_intensity_norm, patch_validation_intensity_norm,patch_training_depth_norm,patch_validation_depth_norm = create_histograms(config, Dir_import, ppp,SBR_mean)

# ---- 2. Create First and Second peak histograms 
level = 4
Hist_1_train, Hist_2_train = second_peak_histogram(Histogram_training_depth_LR_DS, 'dict', level)
Hist_1_val, Hist_2_val = second_peak_histogram(Histogram_validation_depth_LR_DS, 'dict', level)

# ---- 3. Center of Mass -------------------------------------------------------------------

print("Compute Input ...")
Depth_Train = center_of_mass(Hist_1_train, 'dict')
Depth_Train_sec = center_of_mass(Hist_2_train, 'dict')

Depth_Val = center_of_mass(Hist_1_val, 'dict')
Depth_Val_sec = center_of_mass(Hist_2_val, 'dict')

# ---- 4. Upsampling to get Input -------------------------------------------------------------------

Input_train = {}
for patch_idx in range(0, len(Depth_Train)): 
    image = Depth_Train[patch_idx]
    image_up = np.kron(image , np.ones((scale,scale)))
    Input_train[patch_idx] = image_up

Input_val = {}
for patch_idx in range(0, len(Depth_Val)): 
    image = Depth_Val[patch_idx]
    image_up = np.kron(image , np.ones((scale,scale)))
    Input_val[patch_idx] = image_up

Input_train_sec = {}
for patch_idx in range(0, len(Depth_Train_sec)): 
    image = Depth_Train_sec[patch_idx]
    image_up = np.kron(image , np.ones((scale,scale)))
    Input_train_sec[patch_idx] = image_up

Input_val_sec = {}
for patch_idx in range(0, len(Depth_Val_sec)): 
    image = Depth_Val_sec[patch_idx]
    image_up = np.kron(image , np.ones((scale,scale)))
    Input_val_sec[patch_idx] = image_up

# ----5. Upsampling to get Feature 1 ------------------------------------------------------------
list_pool_1 = {}
scale_1 = 2
for patch_idx in range(0, len(Depth_Train)): 
    image = Depth_Train[patch_idx]
    image_up = np.kron(image , np.ones((scale_1,scale_1)))
    list_pool_1[patch_idx] = image_up

list_pool_1_sec = {}
scale_1 = 2
for patch_idx in range(0, len(Depth_Train_sec)): 
    image = Depth_Train_sec[patch_idx]
    image_up = np.kron(image , np.ones((scale_1,scale_1)))
    list_pool_1_sec[patch_idx] = image_up

list_pool_1_val = {}
for patch_idx in range(0, len(Depth_Val)): 
    image = Depth_Val[patch_idx]
    image_up = np.kron(image , np.ones((scale_1,scale_1)))
    list_pool_1_val[patch_idx] = image_up

list_pool_1_val_sec = {}
for patch_idx in range(0, len(Depth_Val_sec)): 
    image = Depth_Val_sec[patch_idx]
    image_up = np.kron(image , np.ones((scale_1,scale_1)))
    list_pool_1_val_sec[patch_idx] = image_up

# ---- 6. Feature 2 ------------------------------------------------------------
list_pool_2 = Depth_Train
list_pool_2_val = Depth_Val

list_pool_2_sec = Depth_Val
list_pool_2_val_sec = Depth_Val_sec

# ---- 7. Feature 3 : DS histogram + center of mass -----------------------------------------
pool_3_hist = {}
pool_3_hist_sec = {}
scale_3 = 2
Nbins = 15
for patch_idx in range(0, len(Histogram_training_depth_LR_DS)): 
    histogram = Histogram_training_depth_LR_DS[patch_idx]
    Nx = histogram.shape[0]
    Ny = histogram.shape[1]
    Hist_LR = np.zeros((int(Nx/scale_3),int(Ny/scale_3),Nbins))
    i_x = 0
    for x in range(0,histogram.shape[0],scale_3):
        i_y = 0
        for y in range(0,histogram.shape[1],scale_3):
            for index_x in range(scale_3):
                for index_y in range(scale_3):
                    Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale_3*scale_3) * histogram[x + index_x, y + index_y, :]
            i_y = i_y + 1
        i_x = i_x + 1
    pool_3_hist[patch_idx], pool_3_hist_sec[patch_idx] = second_peak_histogram(Hist_LR, 'one_image', level)
list_pool_3 = center_of_mass(pool_3_hist, 'dict')
list_pool_3_sec = center_of_mass(pool_3_hist_sec, 'dict')

pool_3_hist_val = {}
pool_3_hist_val_sec = {}
scale_3 = 2
Nbins = 15
for patch_idx in range(0, len(Histogram_validation_depth_LR_DS)): 
    histogram = Histogram_validation_depth_LR_DS[patch_idx]
    Nx = histogram.shape[0]
    Ny = histogram.shape[1]
    Hist_LR = np.zeros((int(Nx/scale_3),int(Ny/scale_3),Nbins))
    i_x = 0
    for x in range(0,histogram.shape[0],scale_3):
        i_y = 0
        for y in range(0,histogram.shape[1],scale_3):
            for index_x in range(scale_3):
                for index_y in range(scale_3):
                    Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale_3*scale_3) * histogram[x + index_x, y + index_y, :]
            i_y = i_y + 1
        i_x = i_x + 1
    pool_3_hist_val[patch_idx], pool_3_hist_val_sec[patch_idx]  = second_peak_histogram(Hist_LR, 'one_image', level)
list_pool_3_val = center_of_mass(pool_3_hist_val, 'dict')
list_pool_3_val_sec = center_of_mass(pool_3_hist_val_sec, 'dict')

# ---- 8. Feature 4 : DS histogram + center of mass -----------------------------------------
pool_4_hist = {}
pool_4_hist_sec = {}
scale_3 = 4
Nbins = 15
for patch_idx in range(0, len(Histogram_training_depth_LR_DS)): 
    histogram = Histogram_training_depth_LR_DS[patch_idx]
    Hist_LR = np.zeros((int(Nx/scale_3),int(Ny/scale_3),Nbins))
    i_x = 0
    for x in range(0,histogram.shape[0],scale_3):
        i_y = 0
        for y in range(0,histogram.shape[1],scale_3):
            for index_x in range(scale_3):
                for index_y in range(scale_3):
                    Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale_3*scale_3) * histogram[x + index_x, y + index_y, :]
            i_y = i_y + 1
        i_x = i_x + 1
    pool_4_hist[patch_idx], pool_4_hist_sec[patch_idx]  = second_peak_histogram(Hist_LR, 'one_image', level)
list_pool_4 = center_of_mass(pool_4_hist, 'dict')
list_pool_4_sec = center_of_mass(pool_4_hist_sec, 'dict')

pool_4_hist_val = {}
pool_4_hist_val_sec = {}
scale_3 = 4
Nbins = 15
for patch_idx in range(0, len(Histogram_validation_depth_LR_DS)): 
    histogram = Histogram_validation_depth_LR_DS[patch_idx]
    Hist_LR = np.zeros((int(Nx/scale_3),int(Ny/scale_3),Nbins))
    i_x = 0
    for x in range(0,histogram.shape[0],scale_3):
        i_y = 0
        for y in range(0,histogram.shape[1],scale_3):
            for index_x in range(scale_3):
                for index_y in range(scale_3):
                    Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale_3*scale_3) * histogram[x + index_x, y + index_y, :]
            i_y = i_y + 1
        i_x = i_x + 1
    pool_4_hist_val[patch_idx], pool_4_hist_val_sec[patch_idx] = second_peak_histogram(Hist_LR, 'one_image', level)
list_pool_4_val = center_of_mass(pool_4_hist_val, 'dict')
list_pool_4_val_sec = center_of_mass(pool_4_hist_val_sec, 'dict')

for scale in [2,4,8,16]:
    Nx = 96
    if scale == 2:
        image_size_2 = int(Nx/scale)
    elif scale == 4:
        image_size_4 = int(Nx/scale)
    elif scale == 8:
        image_size_8 = int(Nx/scale)
    elif scale == 16:
        image_size_16 = int(Nx/scale)

print(str(len(list_pool_1))+' depths for Feature 1 of size '+ str(list_pool_1[0].shape))
print(str(len(list_pool_2))+' depths for Feature 2 of size '+ str(list_pool_2[0].shape))
print(str(len(list_pool_3))+' depths for Feature 3 of size '+ str(list_pool_3[0].shape))
print(str(len(list_pool_4))+' depths for Feature 4 of size '+ str(list_pool_4[0].shape))


print(str(len(list_pool_1_val))+' depths for Feature 1 of size '+ str(list_pool_1_val[0].shape))
print(str(len(list_pool_2_val))+' depths for Feature 2 of size '+ str(list_pool_2_val[0].shape))
print(str(len(list_pool_3_val))+' depths for Feature 3 of size '+ str(list_pool_3_val[0].shape))
print(str(len(list_pool_4_val))+' depths for Feature 4 of size '+ str(list_pool_4_val[0].shape)+'\n')


# ---- 9. Check ppp and SBR levels computed --------------------------------------------------------------------------------------

print('Calculate SBR and ppp ...')
mean_SBR = []
mean_ppp = []
Nimage = len(Histogram_training_depth_LR_DS)
for index in range(Nimage):

    simulated_initial_histogram = np.squeeze(Histogram_training_depth_LR_DS[index])
    Nx_LR = simulated_initial_histogram.shape[0]
    Ny_LR = simulated_initial_histogram.shape[1]
    Nbins = simulated_initial_histogram.shape[2]


    SBR_array = []
    number_zeros = 0
    size_bins_array = []
    b_val_array = []
    for i in range(Nx_LR):
        for j in range(Ny_LR):
            hist_one_pixel = np.squeeze(simulated_initial_histogram[i,j,:])
            pos_max = np.argmax(hist_one_pixel)
            range_center_of_mass = range(max(pos_max-1, 0), min(pos_max+2,Nbins))
            size_bins_array.append(len(range_center_of_mass))
            b = np.median(hist_one_pixel)
            
            if b == 0:
                number_zeros = number_zeros + 1 
            else:
                hist_one_pixel_no_noise =   hist_one_pixel - b * np.ones(Nbins)
                backgound_level = b*len(range_center_of_mass)
                b_val_array.append(b)
                ppp_val = np.sum(hist_one_pixel_no_noise[range_center_of_mass])
                SBR_val = ppp_val/backgound_level
                SBR_array.append(SBR_val)

    SBR_check = np.mean(SBR_array)
    SBR_std = np.std(SBR_array)

    size_bins = np.mean(size_bins_array)
    mean_SBR.append(SBR_check)
    b_val_mean = np.mean(b_val_array)

    # Calculate ppp levels in simulated_initial_histogram 
    ppp_array = []
    for i in range(Nx_LR):
        for j in range(Ny_LR):
            hist_one_pixel = np.squeeze(simulated_initial_histogram[i,j,:])
            pos_max = np.argmax(hist_one_pixel)
            range_center_of_mass = range(max(pos_max-1, 0), min(pos_max+2,Nbins))
            b = np.median(hist_one_pixel)
            hist_one_pixel_no_noise = np.zeros(Nbins)

            hist_one_pixel_no_noise =   hist_one_pixel - b * np.ones(Nbins)
            ppp_val = np.sum(hist_one_pixel_no_noise[range_center_of_mass])
            ppp_array.append(ppp_val)


    ppp_check = np.mean(ppp_array)
    mean_ppp.append(ppp_check)

SBR_final_value = np.mean(mean_SBR)
ppp_final_value = np.mean(ppp_check)
print('SBR_final_value='+str(SBR_final_value))
print('ppp_final_value='+str(ppp_final_value))


# ---- 10. Save --------------------------------------------------------------------------------------

#training
print('Save ...')
index_save = 0
print(len(patch_training_depth_norm))
print(Directory + '\n')
image_size = 96 
image_size_2 = 48
image_size_4 = 24
image_size_8 = 12
image_size_16 = 6
for index_batch in range(0 , len(patch_training_depth_norm) - batch_size , batch_size):
    for index_image in range(0 , batch_size , 1):

        depth_HR    = np.reshape(patch_training_depth_norm[index_batch + index_image] , (1, image_size,image_size))
        intensity   = np.reshape(patch_training_intensity_norm[index_batch + index_image] , (1, image_size,image_size))

        depth_LR = np.reshape(Input_train[index_batch + index_image] , (1, image_size,image_size))
        depth_LR_sec = np.reshape(Input_train_sec[index_batch + index_image] , (1, image_size,image_size))

        pool_1 = np.reshape(list_pool_1[index_batch + index_image] , (1, image_size_2 ,image_size_2))
        pool_2 = np.reshape(list_pool_2[index_batch + index_image] , (1, image_size_4 ,image_size_4))
        pool_3 = np.reshape(list_pool_3[index_batch + index_image] , (1, image_size_8 ,image_size_8))
        pool_4 = np.reshape(list_pool_4[index_batch + index_image] , (1, image_size_16 ,image_size_16))

        pool_1_sec = np.reshape(list_pool_1_sec[index_batch + index_image] , (1, image_size_2 ,image_size_2))
        pool_2_sec = np.reshape(list_pool_2_sec[index_batch + index_image] , (1, image_size_4 ,image_size_4))
        pool_3_sec = np.reshape(list_pool_3_sec[index_batch + index_image] , (1, image_size_8 ,image_size_8))
        pool_4_sec = np.reshape(list_pool_4_sec[index_batch + index_image] , (1, image_size_16 ,image_size_16))

        if index_image == 0:
            batch_depth_HR  = depth_HR
            batch_intensity = intensity
            batch_depth_LR  = depth_LR
            batch_depth_LR_sec = depth_LR_sec
            batch_pool_1    = pool_1
            batch_pool_2    = pool_2
            batch_pool_3    = pool_3
            batch_pool_4    = pool_4  
            batch_pool_1_sec    = pool_1_sec
            batch_pool_2_sec    = pool_2_sec
            batch_pool_3_sec    = pool_3_sec
            batch_pool_4_sec    = pool_4_sec 
            
        else:
            batch_depth_HR  = np.concatenate((batch_depth_HR , depth_HR), axis=0)
            batch_intensity = np.concatenate((batch_intensity , intensity), axis=0)
            batch_depth_LR  = np.concatenate((batch_depth_LR , depth_LR), axis=0)
            batch_depth_LR_sec = np.concatenate((batch_depth_LR_sec , depth_LR_sec), axis=0)
            batch_pool_1    = np.concatenate((batch_pool_1 , pool_1), axis=0)
            batch_pool_2    = np.concatenate((batch_pool_2 , pool_2), axis=0)
            batch_pool_3    = np.concatenate((batch_pool_3 , pool_3), axis=0)
            batch_pool_4    = np.concatenate((batch_pool_4 , pool_4), axis=0)
            batch_pool_1_sec    = np.concatenate((batch_pool_1_sec , pool_1_sec), axis=0)
            batch_pool_2_sec    = np.concatenate((batch_pool_2_sec , pool_2_sec), axis=0)
            batch_pool_3_sec    = np.concatenate((batch_pool_3_sec , pool_3_sec), axis=0)
            batch_pool_4_sec    = np.concatenate((batch_pool_4_sec , pool_4_sec), axis=0)

    dict_HR      = {}
    dict_hist_LR = {}
    dict_i       = {}
    dict_LR      = {}
    dict_LR_sec  = {}
    dict_pool1 = {}
    dict_pool2 = {}
    dict_pool3 = {}
    dict_pool4 = {}
    dict_pool1_sec = {}
    dict_pool2_sec = {}
    dict_pool3_sec = {}
    dict_pool4_sec = {}

    dict_HR['batch_depth_HR'] = batch_depth_HR
    dict_i['batch_intensity'] = batch_intensity
    dict_LR['batch_depth_LR']  = batch_depth_LR
    dict_LR_sec['batch_depth_LR_sec'] = batch_depth_LR_sec
    dict_pool1['batch_pool1'] = batch_pool_1
    dict_pool2['batch_pool2'] = batch_pool_2
    dict_pool3['batch_pool3'] = batch_pool_3
    dict_pool4['batch_pool4'] = batch_pool_4
    dict_pool1_sec['batch_pool1_sec'] = batch_pool_1_sec
    dict_pool2_sec['batch_pool2_sec'] = batch_pool_2_sec
    dict_pool3_sec['batch_pool3_sec'] = batch_pool_3_sec
    dict_pool4_sec['batch_pool4_sec'] = batch_pool_4_sec

    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_depth_label.mat'), dict_HR)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_I_add.mat'), dict_i)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_depth_down.mat'), dict_LR)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_depth_down_2.mat'), dict_LR_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool1.mat'), dict_pool1)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool2.mat'), dict_pool2)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool3.mat'), dict_pool3)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool4.mat'), dict_pool4)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool1_2.mat'), dict_pool1_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool2_2.mat'), dict_pool2_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool3_2.mat'), dict_pool3_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_save)+'_patch_pool4_2.mat'), dict_pool4_sec)
    index_save = index_save + 1

print('Training : '+str(index_save)+' batches')

#validation
index_save = 1
for index_image in range(0,len(patch_validation_depth_norm),1):
    dict_HR      = {}
    dict_i       = {}
    dict_LR      = {}
    dict_pool1 = {}
    dict_pool2 = {}
    dict_pool3 = {}
    dict_pool4 = {}

    dict_HR['depth_label'] = np.reshape(patch_validation_depth_norm[index_image], (96,96))
    dict_i['I_add'] = np.reshape(patch_validation_intensity_norm[index_image], (96,96))
    dict_LR['batch_depth_LR'] = np.reshape(Input_val[index_image] , (96, 96))
    dict_LR_sec['batch_depth_LR_sec'] = np.reshape(Input_val_sec[index_image] , (96, 96))
    dict_pool1['batch_pool_1'] = np.reshape(list_pool_1_val[index_image] , (48 ,48))
    dict_pool2['batch_pool_2'] = np.reshape(list_pool_2_val[index_image] , (24 ,24))
    dict_pool3['batch_pool_3'] = np.reshape(list_pool_3_val[index_image] , (12 ,12))
    dict_pool4['batch_pool_4'] = np.reshape(list_pool_4_val[index_image] , (6 ,6))

    dict_pool1_sec['batch_pool_1_sec'] = np.reshape(list_pool_1_val_sec[index_image] , (48 ,48))
    dict_pool2_sec['batch_pool_2_sec'] = np.reshape(list_pool_2_val_sec[index_image] , (24 ,24))
    dict_pool3_sec['batch_pool_3_sec'] = np.reshape(list_pool_3_val_sec[index_image] , (12 ,12))
    dict_pool4_sec['batch_pool_4_sec'] = np.reshape(list_pool_4_val_sec[index_image] , (6 ,6))

    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_depth_label_test.mat'), dict_HR)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_I_add_test.mat'), dict_i)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_depth_down_test.mat'), dict_LR)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_depth_down_test_2.mat'), dict_LR_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool1_test.mat'), dict_pool1)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool2_test.mat'), dict_pool2)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool3_test.mat'), dict_pool3)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool4_test.mat'), dict_pool4)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool1_test_2.mat'), dict_pool1_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool2_test_2.mat'), dict_pool2_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool3_test_2.mat'), dict_pool3_sec)
    scipy.io.savemat(os.path.join(Directory, str(index_image)+'_patch_pool4_test_2.mat'), dict_pool4_sec)

    index_save = index_save + 1

print('Validation : '+str(index_save)+' batches')
print('Done!')
