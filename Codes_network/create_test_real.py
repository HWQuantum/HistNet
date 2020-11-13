from skimage import io
import os
import scipy
from scipy import misc, ndimage
import scipy.io as sio
import glob
import imageio
import numpy as np
#import skimage.measure
from ops_dataset import *


# ---------------------------------------------------------------------------------
### TO DEFINE
# ---------------------------------------------------------------------------------


Dir = '/home/ar432/HistSR_Net_Repository'
data_type = 'Hammer'
data_type = 'Juggling'


# ---------------------------------------------------------------------------------
### Image index 
# ---------------------------------------------------------------------------------
if data_type == 'Hammer':
    #level = 12 <-level until now
    level = 17 #(Empirically found.) 
    liste_image = [1,9]
elif data_type == 'Juggling':
    level = 7 #(Empirically found.) 
    liste_image = [0,40]

# ---------------------------------------------------------------------------------
### 1. Calibration Matrix
# ---------------------------------------------------------------------------------
calibration = sio.loadmat(os.path.join(Dir,'Real_Data','compensation_frame.mat'))
calibration = 1/15 * calibration['compensation_frame'] 
calibration[0,:]   = calibration[1,:]
calibration[29,:]   = calibration[28,:]
calibration[30,:]   = calibration[28,:]
calibration[31,:]   = calibration[28,:]

calibration_input = np.kron(calibration,np.ones((4 , 4))) # input
calibration_input = np.reshape(calibration_input, (1,128,256,1))

calibration_pool1 = np.kron(calibration,np.ones((2 , 2))) 
calibration_pool1 = np.reshape(calibration_pool1, (1,64,128,1))

calibration_pool2 = np.reshape(calibration, (1,32,64,1))

calibration_pool3 = skimage.measure.block_reduce(calibration, (2,2), np.mean)
calibration_pool3 = np.reshape(calibration_pool3, (1,16,32,1))

calibration_pool4 = skimage.measure.block_reduce(calibration_pool3, (1,2,2,1), np.mean)
calibration_pool4 = np.reshape(calibration_pool4, (1,8,16,1))


# ---------------------------------------------------------------------------------
### 2. Compute Depth, Features and intensity for each image
# ---------------------------------------------------------------------------------
print('Compute Depth and Features...')
data_dir = os.path.join(Dir, 'Real_Data',data_type, 'RAW')

for idx in liste_image:
    h = 128
    w = 256
    c_dim = 1
    scale = 4
    Nbins = 15
    # Intensity 
    intensity = glob.glob(os.path.join(data_dir,str(idx)+'_RGB.mat'))
    intensity = sio.loadmat(intensity[0])['intensity']

    # Input
    hist_input_image   = glob.glob(os.path.join(data_dir,str(idx)+'_hist.mat'))
    histogram_down = sio.loadmat(hist_input_image[0])['histogram']
    histogram_down = histogram_down.reshape([1, int(h/scale), int(w/scale), Nbins])
    histogram_down = np.squeeze(histogram_down)

    histogram_step_1, histogram_step_2  = second_peak_histogram_real(histogram_down, 'one_image', level)
    depth_1 = center_of_mass(histogram_step_1, 'one_image') 
    depth_2 = center_of_mass(histogram_step_2, 'one_image') 
    depth_1 = np.squeeze(depth_1)
    depth_2 = np.squeeze(depth_2)
    
    calibration_input = np.reshape(calibration_input, (128,256))
    depth_1_up = np.kron(depth_1,np.ones((scale , scale))) + calibration_input
    depth_2_up = np.kron(depth_2, np.ones((scale , scale)))

    calibration_input_2 = np.zeros((depth_2_up.shape[0], depth_2_up.shape[1]))

    for i in range(depth_2_up.shape[0]):
        for j in range(depth_2_up.shape[1]):
            if depth_2_up[i,j] == 0:
                calibration_input_2[i,j] = 0
            else :
                calibration_input_2[i,j] = calibration_input[i,j]

    depth_2_up = depth_2_up + calibration_input_2
    
    # -----------------------  Upsample Low resolution Histogram  --------------------------
    histogram_up = np.kron(histogram_down,np.ones((scale , scale , 1)))


    # -----------------------  From Histogram construct Features Input Pyramid  --------------------------
    batch_histogram_down  = np.squeeze(histogram_up)
    Nx, Ny = batch_histogram_down.shape[0], batch_histogram_down.shape[1]
    Nbins = batch_histogram_down.shape[2]
        
    for scale in [2,4,8,16]:
        Hist_LR = np.zeros((int(Nx/scale),int(Ny/scale),Nbins))
        i_x = 0
        for x in range(0,batch_histogram_down.shape[0],scale):
            i_y = 0
            for y in range(0,batch_histogram_down.shape[1],scale):
                for index_x in range(scale):
                    for index_y in range(scale):
                        Hist_LR[i_x, i_y, :] = Hist_LR[i_x, i_y, :] + 1/(scale*scale) * batch_histogram_down[x + index_x, y + index_y, :]
                i_y = i_y + 1
            i_x = i_x + 1
        Depth_LR = center_of_mass(Hist_LR, 'one_image')
        if scale == 2:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_1_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1])
            list_pool_1 = list_pool_1_bec + calibration_pool1
            #print('list_pool_1'+str(list_pool_1.shape))
        elif scale == 4:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_2_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1])
            list_pool_2 = list_pool_2_bec + calibration_pool2
            #print('list_pool_2'+str(list_pool_2.shape))
        elif scale == 8:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_3_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1]) 
            list_pool_3 = list_pool_3_bec + calibration_pool3
            #print('list_pool_3'+str(list_pool_3.shape))
            
        elif scale == 16:
            Nx_2 = int(Nx/scale)
            Ny_2 = int(Ny/scale)
            list_pool_4_bec = np.reshape(Depth_LR, [1 , Nx_2 , Ny_2,1]) 
            list_pool_4 = list_pool_4_bec + calibration_pool4
            #print('list_pool_4'+str(list_pool_4.shape))
    
# ---------------------------------------------------------------------------------
### 3.   Save
# ---------------------------------------------------------------------------------
    print('Save ...')
    save_path = os.path.join(Dir, 'Real_Data',data_type, 'DATA_TEST_frame_'+str(idx))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    sio.savemat(os.path.join(save_path,  "0_Df_down.mat" ),{'I_down':np.squeeze(depth_1_up)})
    sio.savemat(os.path.join(save_path,  "0_Df_down_2.mat" ),{'I_down_2':np.squeeze(depth_2_up)})
    sio.savemat(os.path.join(save_path,  "0_pool1.mat" ),{'list_pool_1':np.squeeze(list_pool_1)})
    sio.savemat(os.path.join(save_path,  "0_pool2.mat" ),{'list_pool_2':np.squeeze(list_pool_2)})
    sio.savemat(os.path.join(save_path,  "0_pool3.mat" ),{'list_pool_3':np.squeeze(list_pool_3)})
    sio.savemat(os.path.join(save_path,  "0_pool4.mat" ),{'list_pool_4':np.squeeze(list_pool_4)})
    sio.savemat(os.path.join(save_path,  "0_init_hist.mat" ),{'histogram_down':np.squeeze(histogram_down)})
    imageio.imwrite(os.path.join(save_path,  "0_RGB.bmp" ), intensity)
    sio.savemat(os.path.join(save_path,  "0_Df.mat" ),{'I_up':np.zeros((h,w))}) # (label inexistent)

