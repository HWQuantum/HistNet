
import glob
import os
import subprocess
import numpy as np



data_dir = '/home/ar432/HistSR_Net_Repository/Real_Data/Hammer_simul_noise'
folder_list = glob.glob(os.path.join(data_dir, 'DATA_TEST_*'))

for folder in folder_list :
    print(folder)
    result_fol = os.path.join(folder, 'results')
    if not os.path.exists(result_fol):
        os.makedirs(result_fol)

    subprocess.run(["python3", "main_hist.py", \
        "--data_path="+str(folder), "--is_train=0",\
            "--config="+str('/home/ar432/DepthSR_Net/Configs/cfg_original_scale4.yaml'),\
                "--checkpoint_dir="+str('/home/ar432/HistSR_Net_Repository/Checkpoint/Checkpoint_ppp4_SBR02'),\
                    "--result_path="+str(result_fol), "--save_parameters=0",\
                        "--loss_type="+str('l1'), "--optimizer_type="+str('Proximal')])


