import glob
import os
import subprocess
import numpy as np
import scipy.io as sio
ppp = 1200
SBR = 2
#index_image = 0
Main_dir = '/home/ar432/HistSR_Net_Repository/'

rmse_tab = []

for level in np.linspace(0,100,101):
    rmse_val = 0

    for index_image = [0,1,2,3,4,5]:
        folder = os.path.join(Main_dir, 'Simulate_data','Simulate_multiple_levels','Middlebury_'+str(index_image)+'_ppp'+str(ppp)+'_SBR'+str(SBR)+'_multiple_level'+str(level),'Results')
        rmse = sio.loadmat(os.path.join(folder, 'parameters.mat'))
        rmse = rmse['rmse']
        rmse = rmse['rmse']
        rmse_val = rmse_val + rmse

    rmse_tab.append(rmse_val/6)

sio.savemat(os.path.join(Main_dir, 'Simulate_data','Simulate_multiple_levels','rmse_tab.mat'), {'rmse_tab':rmse_tab})
