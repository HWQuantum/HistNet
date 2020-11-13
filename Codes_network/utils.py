import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
from PIL import Image  #for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
from skimage import io,data,color
import tensorflow as tf
import imageio

FLAGS = tf.compat.v1.flags.FLAGS


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return imageio.imread(path, pilmode = 'RGB', as_gray = True).astype(np.float)
  else:
    return imageio.imread(path, pilmode = 'RGB').astype(np.float)
 

def get_image_batch(train_list,start_id,end_id):
  target_list = train_list[start_id:end_id]
  input_list  = []

  for pair in target_list:
    input_img_ob = scipy.io.loadmat(pair)
    dlist = [key for key in input_img_ob if not key.startswith('__')]
    input_img = input_img_ob[dlist[0]]
    input_list.append(input_img)

  input_list = np.array(input_list)
  if len(input_list.shape) == 3:
    input_list.resize([end_id-start_id, input_list.shape[1], input_list.shape[2], 1])

  return input_list

def get_image_batch_new(train_list):
  #print(train_list)
  input_batch  = []
  input_img_ob = scipy.io.loadmat(train_list)
  
  dlist=[key for key in input_img_ob if not key.startswith('__')]
  input_img = np.array(input_img_ob[dlist[0]])

  if len(input_img.shape) == 3:
    input_img.resize([input_img.shape[0], input_img.shape[1], input_img.shape[2], 1])
  # import pdb  
  # pdb.set_trace()
  return input_img


def rmse(im1,im2):
  # import pdb  
  # pdb.set_trace()
  diff=np.square(im1.astype(np.float)-im2.astype(np.float))
  diff_sum=np.mean(diff)
  rmse=np.sqrt(diff_sum)
  return rmse    


