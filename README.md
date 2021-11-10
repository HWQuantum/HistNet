
# Robust Super-resolution depth imaging via multi-feature fusion deep network - Code and Models            


Contact: Alice Ruget - ar432@hw.ac.uk

Download the complete folder at 
<https://researchportal.hw.ac.uk/en/datasets/dataset-for-robust-super-resolution-depth-imaging-via-a-multi-fea>. 


This folder contains the code implementation and trained models used for the
project. The following describes how to generate training data and perform the
training, evaluation on the Middlebury test dataset, and evaluation on the
captured results. Packaged with the code, we include the captured
and simulated data. 

This document is structured as follows
1. Generating training Data 
2. Training the network
3. Evaluation on simulated Dataset from Middlebury
4. Evaluation on real Dataset


## 1. Generating Training Data

### Steps
 
Training Data is created from the MPI dataset [1,2]. This dataset is composed 23 RGB-D images of size 436x1024 of the MPI dataset. Median filtering was used to get rid of outliers values (see ./Training_Dataset/Codes/fill_MPI_Sintel_depth.m). The clean depth and intensity images are saved in './Training_Dataset/Raw_data_Middlebury_MPI'.  

1. Edit the noise levels (ppp and SBR) in './Training_Dataset/main_creation_dataset.py'.
(The realistic scenario of the paper corresponds to ppp=1200 and SBR=2. The extreme scenario corresponds to ppp=4 and SBR=0.02.)

2. Run 
```python
python3 './Training_Dataset/Codes/main_creation_dataset.py'
```
The datasets will be saved in ./Training_Dataset/Dataset under the names DATA_intensity_1200_SBR_2 for the realistic scenario and DATA_intensity_4_SBR_0_02 for the extreme scenario.


## 2. Training

### Steps
The training has already been performed for the two noise scenarios presented in the paper. The checkpoints are saved in ./Checkpoint (Checkpoint_ppp4_SBR02 for the extreme scenario and Checkpoint_ppp1200_SBR2 for the realistic scenario).

To re-create the results of the training, see the following
1. Install required python packages listed in './requirements.txt' 
2. Edit the paths in the following command according to the noise scenario and run :

```python
python3 ./Codes_Network/main_hist.py 
--data_path='./Training_Dataset/Dataset/DATA_intensity_*ppp*_SBR_*SBR*' --is_train='1'
--config='./Config/config.yaml' --checkpoint_dir = './Checkpoint/*checkpoint_name*' --result_path='./Results_Training/ppp*ppp*_SBR_*SBR*' --save_parameters='1' --loss_type='l1' 
--optimizer_type='Proximal'
```

### Example 
For example, to run the training for the 'realistic' noise scenario:
```python
python3 ./Codes_Network/main_hist.py --data_path='./Training_Dataset/Dataset/DATA_intensity_1200_SBR_2' --is_train='1' --config='./Config/config.yaml' --checkpoint_dir='./Checkpoint/Checkpoint_ppp1200_SBR2' --result_path='/Results_Training/ppp1200_SBR2' --save_parameters='1' --loss_type='l1' --optimizer_type='Proximal'
```



## 3. Evaluation on Simulated Middlebury Dataset
Scenes from the Middlebury Stereo dataset [3,4] were filtered with Median filtering and saved in './Simulate_data/RAW'. 
The following steps 1 and 2 simulate SPAD measurements and the corresponding inputs to the network with a specific noise scenario. The simulated data created by ./Codes/create_test_synthetic.py appear in a folder in ./Simulate_data/Middlebury_*index_image*_ppp*ppp*_SBR*SBR*/data. The index of the images from the paper are 0 and 5. The noise levels presented in the paper are (ppp=1200, SBR =2) and (ppp=4, SBR=0.2).

Step 3 runs the data through the network. Results of each scene are saved in './Simulate_data/*data_name*/Results/parameters.mat.

### Steps

1. Edit image's index, ppp level and SBR levels in ./Codes/create_test_synthetic.py. 
2. Run ```python3 ./Codes_Network/create_test_synthetic.py```
3. Edit the arguments and run : 
```python
python3 ./Codes_Network/main_hist.py 
--data_path='./Simulate_data/*data_name*/data' --is_train='0'
--config='./Config/config.yaml' --checkpoint_dir = './Checkpoint/*checkpoint_name*' --result_path='./Simulate_data/*data_name*/results' --save_parameters='1' --loss_type='l1' 
--optimizer_type='Proximal'
```

### Example 
To reproduce the results on Art scene (index_image=0) of the paper for the noise levels (ppp=4, SBR=0.2), run the following command in the folder HistSR_Net_Repository

```python
python3 ./Codes_Network/main_hist.py --data_path='./Simulate_data/Middlebury_0_ppp4_SBR0.02/data' --is_train='0' --config='./Config/config.yaml' --checkpoint_dir='/home/ar432/HistSR_Net_Repository/Checkpoint/Checkpoint_ppp4_SBR02' --result_path='./Simulate_data/Middlebury_0_ppp4_SBR0.02/Results' --save_parameters='1' --loss_type='l1' --optimizer_type='Proximal'
```

## 4. Evaluation on Real Dataset 
The raw data captured by the SPAD is in ./real_data/*data_type*/RAW with *data_type* being either hammer or juggling. 
Following steps 1-3 (code create_test_real.py) computes the input of HistNet from these raw data which appear in a folder called DATA_TEST in ./real_data/*data_type*/. Step 4 (code main_hist.py) computes the reconstruction in parameters.mat in result_path.

### Steps
1. Edit Directory in ./Codes/create_test_real.py 
2. Choose data_type between Hammer or Juggling in ./Codes_Network/create_test_real.py 
3. Run ./Codes_Network/create_test_real.py to create the input of the network from the Raw data
4. To test the network, run ./Codes/main_hist.py by specifying the following parser arguments : 

```python
python3 ./Codes_Network/main_hist.py
--data_path='./Real_Data/*data_type*/DATA_TEST_frame_*index_frame*'
--is_train='0'
--config='./Config/config.yaml'
--checkpoint_dir='./Checkpoint/Checkpoint_ppp1200_SBR2'
--result_path='./Real_Data/*data_type*/DATA_TEST_frame_*index_frame*/results'
--save_parameters='1'
--loss_type='l1' 
--optimizer_type='Proximal'
```

### Examples 


Example: To reconstruct frame 9 of the hammer data, run :

```python
python3 ./Codes_Network/main_hist.py --data_path='./Real_Data/Hammer/DATA_TEST_frame_9' --is_train='0' --config='./Config/config.yaml' --checkpoint_dir='./Checkpoint/Checkpoint_ppp1200_SBR2' --result_path='./Real_Data/Hammer/DATA_TEST_frame_9/results' --save_parameters='1' --loss_type='l1' --optimizer_type='Adagrad'
```

To reconstruct frame 40 of the juggling data, run : 
```python
python3 ./Codes_Network/main_hist.py --data_path='./Real_Data/Juggling/DATA_TEST_frame_40' --is_train='0' --config='./Config/config.yaml' --checkpoint_dir='./Checkpoint/Checkpoint_ppp1200_SBR2' --result_path='./Real_Data/Juggling/DATA_TEST_frame_40/results' --save_parameters='1' --loss_type='l1' --optimizer_type='Adagrad'
```
## References 

[1] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black, “A naturalistic open source movie for optical flow evaluation,” inEuropean Conf. on Computer Vision(ECCV),A. Fitzgibbon et al. (Eds.), ed. (Springer-Verlag, 2012), Part IV, LNCS 7577, pp. 611–625.

[2] J. Wulff, D. J. Butler, G. B. Stanley, and M. J. Black, “Lessons and insights from creating a synthetic optical flow benchmark,” inECCV Workshop onUnsolved Problems in Optical Flow and Stereo Estimation,A. Fusiello et al. (Eds.), ed. (Springer-Verlag, 2012), Part II, LNCS 7584, pp. 168–177

[3] D. Scharstein and C. Pal. Learning conditional random fields for stereo.
In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007.

[4] H. Hirschmüller and D. Scharstein. Evaluation of cost functions for stereo matching.
In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007.


