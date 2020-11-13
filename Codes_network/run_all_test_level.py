import glob
import os
import subprocess
import numpy as np

Main_dir = '/home/ar432/HistSR_Net_Repository/'
case=1
if case ==1:
#for case in [1,2]:
    if case ==1:
        ppp = 1200
        SBR = 2
        SBR_check = '2'
    if case == 2:
        ppp = 4
        SBR = 0.02
        SBR_check = '02'

    for index_image  in [0,1,2,3,4,5]:
    #index_image = 0
    #if index_image ==0:
        for level in np.linspace(0, 50,51):
            folder = os.path.join(Main_dir, 'Simulate_data','Simulate_multiple_levels','Middlebury_'+str(index_image)+'_ppp'+str(ppp)+'_SBR'+str(SBR)+'_multiple_level'+str(level))

            result_folder = os.path.join(folder, 'Results')
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            
            data_folder = os.path.join(folder, 'data')
            checkpoint_folder = os.path.join(Main_dir,'Checkpoint', 'Checkpoint_ppp'+str(ppp)+'_SBR'+SBR_check)


            subprocess.run(["python3", "main_hist.py", \
                "--data_path="+str(data_folder), "--is_train=0",\
                    "--config="+str('/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml'),\
                        "--checkpoint_dir="+str(checkpoint_folder),\
                            "--result_path="+str(result_folder), "--save_parameters=1",\
                                "--loss_type="+str('l1'), "--optimizer_type="+str('Proximal')])


