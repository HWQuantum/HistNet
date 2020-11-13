from model_network_hist import SRCNN
import numpy as np
import tensorflow as tf

import glob
import pprint
import os
from scipy import misc
import scipy.io as sio
import argparse
import yaml
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config_sess = tf.compat.v1.ConfigProto()
config_sess.gpu_options.allow_growth = True 
config_sess.allow_soft_placement = True 

parser = argparse.ArgumentParser('')
parser.add_argument('--data_path', type=str)
parser.add_argument('--is_train', type=str)
parser.add_argument('--result_path', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--save_parameters', type=bool)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--optimizer_type', type=str)

args = parser.parse_args()

data_path = args.data_path
is_train = args.is_train 
results_path = args.result_path
config = yaml.load(open(args.config, 'r'))
checkpoint_path = args.checkpoint_dir
save_parameters = args.save_parameters
loss_type = args.loss_type
optimizer_type = args.optimizer_type
flags = tf.compat.v1.flags

flags.DEFINE_integer("batch_size", config['batch_size'], "The size of batch images [128]")
flags.DEFINE_integer("image_size",config['image_size'] , "The size of image to use [33]")
flags.DEFINE_integer("label_size", config['label_size'], "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", config['learning_rate'], "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", config['c_dim'], "Dimension of image color. [3]")
flags.DEFINE_integer("h0", config['h0'], "Dimension of image h. [230]")
flags.DEFINE_integer("w0", config['w0'], "Dimension of image w. [310]")
flags.DEFINE_integer("scale", config['scale'], "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", config['stride'], "The size of stride to apply input image [14]")
flags.DEFINE_boolean("save_parameters", save_parameters, "True for saving the parameters of network")
flags.DEFINE_string("data_path", data_path, "The Path of Data (test or train)")
flags.DEFINE_string("checkpoint_dir", checkpoint_path, "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", results_path, "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", is_train , "True for training, False for testing [True]")
flags.DEFINE_string("results_path", results_path, "Path of results = parameters + sample")
flags.DEFINE_string("loss_type", loss_type, "type of loss = 'l2' or 'l1'")
flags.DEFINE_string("optimizer_type", optimizer_type, "type optimizer_type loss = 'Adam' or 'Proxi'")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  # ----------------------------------------------------------------     
  # -----------------------  Training  ------------------------------
  # ----------------------------------------------------------------
  if FLAGS.is_train:
    with tf.compat.v1.Session(config=config_sess) as sess:
      srcnn = SRCNN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  h0=FLAGS.h0,
                  w0=FLAGS.w0,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  loss_type=FLAGS.loss_type,
                  optimizer_type=FLAGS.optimizer_type)
      srcnn.train(FLAGS)

  else:
    # ----------------------------------------------------------------     
    # -----------------------  Testing  ------------------------------
    # ----------------------------------------------------------------
    data_dir = data_path      
    rgb_input_list =glob.glob(os.path.join(data_dir,'*_RGB.bmp'))
    image_test=[]
    RMSE_tab = []

    
    for ide in range(0,len(rgb_input_list)):
      image_test.append(np.float32(imageio.imread(rgb_input_list[ide])) / 255)
    
    for idx in range(0,len(rgb_input_list)):
      depth_input_down_image   = glob.glob(os.path.join(data_dir,str(idx)+'_Df_down.mat'))
      depth_input_down_image_2 = glob.glob(os.path.join(data_dir,str(idx)+'_Df_down_2.mat'))
      depth_label_list_image   = glob.glob(os.path.join(data_dir,str(idx)+'_Df.mat'))
      rgb_input_list_image     = glob.glob(os.path.join(data_dir,str(idx)+'_RGB.bmp'))
      pool1_list_image         = glob.glob(os.path.join(data_dir,str(idx)+'_pool1.mat'))
      pool2_list_image         = glob.glob(os.path.join(data_dir,str(idx)+'_pool2.mat'))
      pool3_list_image         = glob.glob(os.path.join(data_dir,str(idx)+'_pool3.mat'))
      pool4_list_image         = glob.glob(os.path.join(data_dir,str(idx)+'_pool4.mat'))

     
      print('index_image = '+str(idx))
      print(depth_input_down_image)
      print(depth_label_list_image)
      print(rgb_input_list_image)
      
      depth_up = sio.loadmat(depth_label_list_image[0])['I_up']
      image_path = results_path
      image_path = os.path.join(image_path, str(idx)+"_up.png" )
      [IMAGE_HEIGHT,IMAGE_WIDTH] = image_test[idx].shape
      print(IMAGE_HEIGHT)
      print(IMAGE_WIDTH)
      with tf.compat.v1.Session() as sess:
        srcnn = SRCNN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  h0=IMAGE_HEIGHT,
                  w0=IMAGE_WIDTH,
                  c_dim=1, 
                  i=idx,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  test_dir=rgb_input_list_image[0],
                  test_depth=depth_input_down_image[0],
                  test_depth_2=depth_input_down_image_2[0],
                  test_label=depth_label_list_image[0],
                  loss_type=FLAGS.loss_type,
                  optimizer_type=FLAGS.optimizer_type, 
                  pool1 = pool1_list_image[0],
                  pool2 = pool2_list_image[0],
                  pool3 = pool3_list_image[0],
                  pool4 = pool4_list_image[0]
                  )
        recon = srcnn.train(FLAGS)
        
        if not os.path.exists(os.path.join(results_path,'Reconstructions')):
            os.makedirs(os.path.join(results_path,'Reconstructions'))
        sio.savemat(os.path.join(results_path,'Reconstructions',str(idx)+'recon.mat'), {'recon':recon})
if __name__ == '__main__':
 tf.compat.v1.app.run()
