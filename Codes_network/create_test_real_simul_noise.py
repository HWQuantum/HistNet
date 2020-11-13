from skimage import io
import os
from PIL import Image
import scipy
from scipy import misc, ndimage
import scipy.io as sio
import glob
import imageio
import numpy as np
from ops_dataset import (
    create_noise,
    create_hist,
    center_of_mass,
    second_peak_histogram
)
def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return imageio.imread(path, pilmode = 'RGB', as_gray = True).astype(np.float)
  else:
    return imageio.imread(path, pilmode = 'RGB').astype(np.float)

### --- To Define 
# Scenarios of paper : 1/ ppp = 1200, SBR = 2 ; 2/ ppp = 4, SBR = 0.02
# The images in the paper are of index_image = 0,4 and 5.
#Init_Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/'

Init_Dir = '/home/ar432/HistSR_Net_Repository'
ppp = 1200
SBR = 2
#ppp = 4
#SBR = 0.02
index_image = 0 

### --- Import data 
print('Import data ...')
Dir = os.path.join(Init_Dir, 'Real_Data')


liste_image = range(48)
for index in liste_image:
    folder = os.path.join(Dir, 'Hammer', 'DATA_TEST_frame_'+str(index))
    folder_intensity = os.path.join(folder,'0_RGB.bmp')
    depth = sio.loadmat(os.path.join(folder,'0_Df_down.mat'))['I_down']
    intensity = imread(folder_intensity, is_grayscale=True)/255

    
    I_up = np.squeeze(depth)
    intensity_image = np.squeeze(intensity)

    ### --- Crop images modulo 16 
    h, w = I_up.shape
    h = h - np.mod(h, 16)
    w = w - np.mod(w, 16)
    I_up = I_up[0:h, 0:w]
    intensity_image = intensity_image[0:h, 0:w]

    ### --- Normalize 
    print('Normalize  ...')
    min_up , max_up  = np.min(I_up), np.max(I_up)
    I_up = (I_up- min_up)/(max_up-min_up)
    min_i, max_i = np.min(intensity_image), np.max(intensity_image)
    intensity_image = (intensity_image-min_i)/(max_i-min_i)


    ### --- Create Histograms 
    print('Create Histograms  ...')
    patch_depth_LR_norm = {}
    patch_depth_LR_norm[0] = I_up
    patch_intensity_norm = {}
    patch_intensity_norm[0] = intensity_image
    mean_val= np.mean(intensity_image)
    patch_histogram = create_hist(patch_depth_LR_norm, patch_intensity_norm, ppp)
    print(patch_histogram[0].shape)

    ### --- Create Noisy Histograms 
    print('Create Noisy Histograms  ...')
    type_background = 'constant_SBR'
    patch_histogram = create_noise(patch_histogram, SBR, type_background)
    histogram = patch_histogram[0]
    before_downsampling_histogram = histogram

    ### --- Create HR intensity 
    print('Create HR intensity  ...')
    intensity_image = np.sum(histogram, 2)

    ### --- Normalize intensity again 
    print('Normalize intensity  ...')
    min_i, max_i = np.min(intensity_image), np.max(intensity_image)
    intensity_image = (intensity_image - min_i)/(max_i - min_i)


    ### --- Downsample Histograms 
    print('Downsample Histograms  ...')
    Nx = histogram.shape[0]
    Ny = histogram.shape[1]
    Nbins = histogram.shape[2]
    scale = 4
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

    histogram = Hist_LR
    simulated_initial_histogram = Hist_LR


    ## --- Create first and second peak Histogram 
    histogram_step_1, histogram_step_2  = second_peak_histogram(histogram)

    depth_1 = center_of_mass(histogram, 'one_image') 
    depth_2 = center_of_mass(histogram_step_2, 'one_image') 

    depth = depth_1

    ### --- Upsampling to get Input 
    print('Input ...')
    scale = 4
    depth_up_1 = np.kron(depth_1 , np.ones((scale,scale)))
    depth_up_2 = np.kron(depth_2 , np.ones((scale,scale)))
    depth_up = depth_up_1

    ### ---  Upsampling to get Feature 1 
    print('Feature 1 ...')
    scale_1 = 2
    image_up = np.kron(depth , np.ones((scale_1,scale_1)))
    list_pool_1 = image_up


    ### --- Feature 2 
    print('Feature 2 ...')
    list_pool_2 = depth


    ### --- Feature 3 
    print('Feature 3 ...')
    scale_3 = 2
    Nbins = 15
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
    list_pool_3 = center_of_mass(Hist_LR, 'one_image')


    ### --- Feature 4 
    print('Feature 4 ...')
    scale_3 = 4
    Nbins = 15
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
    list_pool_4 = center_of_mass(Hist_LR, 'one_image')



    # ### --- Calculate SBR in simulated_initial_histogram 
    print('Calculate SBR and ppp ...')

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
    print('SBR'+str(SBR_check))


    # Calculate ppp levels in simulated_initial_histogram 
    ppp_array = []
    for i in range(Nx_LR):
        for j in range(Ny_LR):
            hist_one_pixel = np.squeeze(simulated_initial_histogram[i,j,:])
            pos_max = np.argmax(hist_one_pixel)
            range_center_of_mass = range(max(pos_max-1, 0), min(pos_max+2,Nbins))
            b = np.median(hist_one_pixel)
            hist_one_pixel_no_noise = np.zeros(Nbins)
            #for t in range(Nbins):
                # hist_one_pixel_no_noise[t] = np.max(hist_one_pixel[t] - b ,0)
            hist_one_pixel_no_noise =   hist_one_pixel - b * np.ones(Nbins)
            ppp_val = np.sum(hist_one_pixel_no_noise[range_center_of_mass])
            ppp_array.append(ppp_val)


    ppp_check = np.mean(ppp_array)

    print('ppp'+str(ppp_check))


    ### --- Save 
    print('Save ...')


    save_path = os.path.join(Dir, 'Hammer_simul_noise')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'DATA_TEST_frame_'+str(index))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)


    sio.savemat(os.path.join(save_path,  "initial_histogram.mat" ),{'initial_histogram':np.squeeze(simulated_initial_histogram)})
    sio.savemat(os.path.join(save_path,  "0_Df_down.mat" ),{'I_down':np.squeeze(depth_up)})
    sio.savemat(os.path.join(save_path,  "0_Df_down_2.mat" ),{'I_down_2':np.squeeze(depth_up_2)})
    sio.savemat(os.path.join(save_path,  "0_Df.mat" ),{'I_up':np.squeeze(I_up)})
    sio.savemat(os.path.join(save_path,  "0_pool1.mat" ),{'list_pool_1':np.squeeze(list_pool_1)})
    sio.savemat(os.path.join(save_path,  "0_pool2.mat" ),{'list_pool_2':np.squeeze(list_pool_2)})
    sio.savemat(os.path.join(save_path,  "0_pool3.mat" ),{'list_pool_3':np.squeeze(list_pool_3)})
    sio.savemat(os.path.join(save_path,  "0_pool4.mat" ),{'list_pool_4':np.squeeze(list_pool_4)})
    imageio.imwrite(os.path.join(save_path,  "0_RGB.bmp" ), intensity_image)
    sio.savemat(os.path.join(save_path,  "parameters_noise.mat" ),{'SBR':SBR, 'ppp':ppp})

