from skimage import io,data,color
import time
import os
import matplotlib.pyplot as plt
from ops import * 
import numpy as np
import tensorflow as tf
import scipy.io as sio
import skimage.filters.rank as sfr
import subprocess
from utils import *


class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=128,
               label_size=128, 
               batch_size=32,
               c_dim=1, 
               i = 0,
               h0=None,
               w0=None,
               checkpoint_dir=None, 
               sample_dir=None,
               test_dir=None,
               test_depth=None,
               test_depth_2=None,
               test_label=None,
               loss_type=None,
               optimizer_type=None, 
               pool1 = None, 
               pool2 = None, 
               pool3 = None, 
               pool4 = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)

    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.h = h0
    self.w = w0
    self.c_dim = c_dim
    self.i=i
    self.test_dir=test_dir
    self.test_depth=test_depth
    self.test_depth_2=test_depth_2
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.test_label = test_label  
    self.loss_type = loss_type  
    self.optimizer_type = optimizer_type
    self.pool1 = pool1
    self.pool2 = pool2
    self.pool3 = pool3
    self.pool4 = pool4
    self.build_model()

  def build_model(self):
    self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1], name='images')
    self.images_2 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1], name='images_2')
    self.labels = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, 1], name='labels')
    self.I_add = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='I_add')
    self.pool1_input = tf.compat.v1.placeholder(tf.float32, [self.batch_size, int(self.image_size/2), int(self.image_size/2), 1], name='pool1')
    self.pool2_input = tf.compat.v1.placeholder(tf.float32, [self.batch_size, int(self.image_size/4), int(self.image_size/4), 1], name='pool2')
    self.pool3_input = tf.compat.v1.placeholder(tf.float32, [self.batch_size, int(self.image_size/8), int(self.image_size/8), 1], name='pool3')
    self.pool4_input = tf.compat.v1.placeholder(tf.float32, [self.batch_size, int(self.image_size/16), int(self.image_size/16), 1], name='pool4')


    self.depth_test = tf.compat.v1.placeholder(tf.float32, [1, self.h, self.w, 1], name='images_test')
    self.depth_test_2 = tf.compat.v1.placeholder(tf.float32, [1, self.h, self.w, 1], name='images_test_2')
    self.I_add_test = tf.compat.v1.placeholder(tf.float32, [1, self.h, self.w, self.c_dim], name='I_add_test')
    self.pool1_input_test = tf.compat.v1.placeholder(tf.float32, [1, int(self.h/2), int(self.w/2), 1], name='pool1_test')
    self.pool2_input_test = tf.compat.v1.placeholder(tf.float32, [1, int(self.h/4), int(self.w/4), 1], name='pool2_test')
    self.pool3_input_test = tf.compat.v1.placeholder(tf.float32, [1, int(self.h/8), int(self.w/8), 1], name='pool3_test')
    self.pool4_input_test = tf.compat.v1.placeholder(tf.float32, [1, int(self.h/16), int(self.w/16), 1], name='pool4_test')


    if self.i < 1 :
      self.pred, self.dictionary, self.residual_map = self.model()
    #else:
    self.pred_test, self.dictionary_test = self.model_test()

    if self.test_dir is None:
      if self.loss_type =='l2':
        self.loss_mse = tf.reduce_mean(tf.square(self.labels - self.pred))

      elif self.loss_type =='l1':
        self.loss_mse = tf.reduce_mean(tf.math.abs(self.labels - self.pred))

      self.loss = self.loss_mse
      self.stddev = tf.math.reduce_std(tf.square(self.labels - self.pred))

    self.saver = tf.compat.v1.train.Saver(max_to_keep=0)




    
  def train(self, config):
    # ----------------------------------------------------------------     
    # -----------------------  Training  -----------------------------
    # ----------------------------------------------------------------
    
    if config.is_train:    
      # --------------------- Import Data --------------------- 
      data_dir = config.data_path
      depth_input_down_list   = sorted(glob.glob(os.path.join(data_dir,'*_patch_depth_down.mat')))
      depth_input_down_list_2 = sorted(glob.glob(os.path.join(data_dir,'*_patch_depth_down_2.mat')))
      depth_label_list        = sorted(glob.glob(os.path.join(data_dir,'*_patch_depth_label.mat')))
      rgb_input_list          = sorted(glob.glob(os.path.join(data_dir,'*_patch_I_add.mat')))
      depth_pool1             = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool1.mat')))
      depth_pool2             = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool2.mat')))
      depth_pool3             = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool3.mat')))
      depth_pool4             = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool4.mat')))

    
      seed=545
      np.random.seed(seed)
      np.random.shuffle(depth_input_down_list)
      np.random.seed(seed)
      np.random.shuffle(depth_input_down_list_2)
      np.random.seed(seed)
      np.random.shuffle(depth_label_list)
      np.random.seed(seed)
      np.random.shuffle(rgb_input_list)
      np.random.seed(seed)
      np.random.shuffle(depth_pool1)
      np.random.seed(seed)
      np.random.shuffle(depth_pool2)
      np.random.seed(seed)
      np.random.shuffle(depth_pool3)
      np.random.seed(seed)
      np.random.shuffle(depth_pool4)

      depth_input_down_list_test  = sorted(glob.glob(os.path.join(data_dir,'*_patch_depth_down_test.mat')))
      depth_input_down_list_test_2= sorted(glob.glob(os.path.join(data_dir,'*_patch_depth_down_test_2.mat')))
      depth_label_list_test       = sorted(glob.glob(os.path.join(data_dir,'*_patch_depth_label_test.mat')))
      rgb_input_list_test         = sorted(glob.glob(os.path.join(data_dir,'*_patch_I_add_test.mat')))
      depth_pool1_test            = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool1_test.mat')))
      depth_pool2_test            = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool2_test.mat')))
      depth_pool3_test            = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool3_test.mat')))
      depth_pool4_test            = sorted(glob.glob(os.path.join(data_dir,'*_patch_pool4_test.mat')))

      # --------------------- Define optimizer ---------------------
      if self.optimizer_type =='Adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(config.learning_rate,0.9)
      elif self.optimizer_type =='Proximal':
        optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(1e-1, initial_accumulator_value=0.1, l1_regularization_strength=0.0, \
           l2_regularization_strength=0.0, use_locking=False, name='ProximalAdagrad')

      self.grads_and_vars = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.minimize(self.loss)

    # --------------------- Download Checkpoint and initialize variables ---------------------
      tf.compat.v1.initialize_all_variables().run()
      counter = 0
      start_time = time.time()
      print(self.checkpoint_dir)
      if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")
        
    # -----------------------  Train  ----------------------------------------------------------
      print("Training...")
      loss_training   = []
      loss_validation = []

      start_time = time.time()
      for ep in range(2000):
        batch_idxs=len(depth_input_down_list)

        for idx in range(0,batch_idxs):
          batch_depth_down    = get_image_batch_new(depth_input_down_list[idx])#/15
          batch_depth_down_2  = get_image_batch_new(depth_input_down_list_2[idx])#/15
          batch_depth_labels  = get_image_batch_new(depth_label_list[idx])
          batch_I_add         = get_image_batch_new(rgb_input_list[idx])#/255
          batch_pool1         = get_image_batch_new(depth_pool1[idx])#/15
          batch_pool2         = get_image_batch_new(depth_pool2[idx])#/15
          batch_pool3         = get_image_batch_new(depth_pool3[idx])#/15
          batch_pool4         = get_image_batch_new(depth_pool4[idx])#/15


          _, err, dictionary = self.sess.run([self.train_op, self.loss,self.dictionary], \
            feed_dict={self.images: batch_depth_down,  self.images_2: batch_depth_down_2, self.labels: batch_depth_labels,self.I_add:batch_I_add,\
              self.pool1_input: batch_pool1, self.pool2_input:batch_pool2, self.pool3_input:batch_pool3, self.pool4_input:batch_pool4})
          counter += 1
          loss_training.append(err)
          dictionary['loss_training'] = loss_training

       
          # ------ Validation  ------ 
          if idx == batch_idxs-1:
            dictionary_validation = {}
            batch_test_idxs = len(depth_input_down_list_test) // config.batch_size
            err_test =  np.ones(batch_test_idxs)

            for idx_test in range(0,batch_test_idxs):
                batch_depth_down_val    = get_image_batch(depth_input_down_list_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)#/15
                batch_depth_down_val_2  = get_image_batch(depth_input_down_list_test_2, idx_test*config.batch_size , (idx_test+1)*config.batch_size)#/15
                batch_depth_labels_val  = get_image_batch(depth_label_list_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)
                batch_I_add_val         = get_image_batch(rgb_input_list_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size) #/255
                batch_pool1_val         = get_image_batch(depth_pool1_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)#/15
                batch_pool2_val         = get_image_batch(depth_pool2_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)#/15
                batch_pool3_val         = get_image_batch(depth_pool3_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)#/15
                batch_pool4_val         = get_image_batch(depth_pool4_test, idx_test*config.batch_size , (idx_test+1)*config.batch_size)#/15
                
                err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: batch_depth_down_val, self.images_2: batch_depth_down_val_2, self.labels: batch_depth_labels_val,self.I_add:batch_I_add_val,\
                  self.pool1_input: batch_pool1_val, self.pool2_input:batch_pool2_val, self.pool3_input:batch_pool3_val, self.pool4_input:batch_pool4_val})    
                dictio =  self.sess.run(self.dictionary,feed_dict={self.images: batch_depth_down_val, self.images_2: batch_depth_down_val_2, self.labels: batch_depth_labels_val,self.I_add:batch_I_add_val,\
                  self.pool1_input: batch_pool1_val, self.pool2_input:batch_pool2_val, self.pool3_input:batch_pool3_val, self.pool4_input:batch_pool4_val}) 
                
                if idx_test == 0:
                  dictionary_validation = dictio

            loss_validation.append(np.mean(err_test))
            dictionary_validation['loss_validation'] = loss_validation

            print("Validation-------Epoch: [%2d], step: [%2d], time: [%4.4f], loss_validation: [%.8f]" \
                % ((ep+1), counter, time.time()-start_time, np.mean(err_test))) 
            
            # ------ Saving Parameters and Checkpoint  ------
            if ep%10==0:
                self.save(config.checkpoint_dir, counter)
                sio.savemat(os.path.join(config.results_path, 'parameters_validation_ep_'+str(ep)+'.mat'), dictionary_validation)
                sio.savemat(os.path.join(config.results_path, 'parameters_training_ep_'+str(ep)+'.mat'), dictionary)
                print('Parameters Saved')

      print('[*][*][*] Training time', time.time() - start_time)



    # ----------------------------------------------------------------     
    # -----------------------  Testing  ------------------------------
    # ----------------------------------------------------------------
    else:
      # -----------------------  Import Data  --------------------------
      
      # Intensity
      I_add_input_test = imread(self.test_dir, is_grayscale=True)/255   
      image_path = os.path.join(config.sample_dir, str(self.i)+"_rgb.png" )
      I_add_input_test = I_add_input_test.reshape([1,self.h,self.w,self.c_dim])

      # Label
      depth_label = sio.loadmat(self.test_label)['I_up']
      max_label=np.max(depth_label)
      min_label=np.min(depth_label)
      Nx = depth_label.shape[0]
      Ny = depth_label.shape[1]

      # Input
      depth_down = sio.loadmat(self.test_depth)['I_down'].astype(np.float)
      depth_down = np.reshape(depth_down, (1, Nx, Ny, 1))

      depth_down_2 = sio.loadmat(self.test_depth_2)['I_down_2'].astype(np.float)
      depth_down_2 = np.reshape(depth_down_2, (1, Nx, Ny, 1))

      list_pool_1 = sio.loadmat(self.pool1)['list_pool_1'].astype(np.float)
      list_pool_1 = np.reshape(list_pool_1, (1, int(Nx/2), int(Ny/2), 1))
      list_pool_2 = sio.loadmat(self.pool2)['list_pool_2'].astype(np.float)
      list_pool_2 = np.reshape(list_pool_2, (1, int(Nx/4), int(Ny/4), 1))
      list_pool_3 = sio.loadmat(self.pool3)['list_pool_3'].astype(np.float)
      list_pool_3 = np.reshape(list_pool_3, (1, int(Nx/8), int(Ny/8), 1))
      list_pool_4 = sio.loadmat(self.pool4)['list_pool_4'].astype(np.float)
      list_pool_4 = np.reshape(list_pool_4, (1, int(Nx/16), int(Ny/16), 1))


      # --------------------- Download Checkpoint and initialize variables ---------------------
      #print(tf.compat.v1.global_variables(scope=None))   
      tf.compat.v1.initialize_all_variables().run()
      counter = 0
      start_time = time.time()
      print(self.checkpoint_dir)
      if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
      else:
        print(" [!] Load failed...")

      # -------------------------- Test --------------------------------------------------------
      print("Testing...")
      tm = time.time()
 
      result, dictionary = self.sess.run([self.pred_test,self.dictionary_test], feed_dict= {self.depth_test:depth_down ,self.depth_test_2:depth_down_2, self.I_add_test:I_add_input_test, \
         self.pool1_input_test:list_pool_1, self.pool2_input_test:list_pool_2, self.pool3_input_test:list_pool_3,self.pool4_input_test:list_pool_4})
    
      print('Rec time ', time.time() - tm)

      result = np.squeeze(result)
      depth_label = np.squeeze(depth_label)
      rmse_value = rmse(depth_label,result) 
      print("rmse: [%f]" % rmse_value)
      init_image = np.squeeze(depth_down)
      init_rmse = rmse(depth_label, init_image)   
      print("initial rmse: [%f]" % init_rmse)
      
      dictionary['rmse']      = {'init_rmse' : init_rmse, 'rmse' : rmse_value}
      dictionary['depth_label']= depth_label

      if config.save_parameters:     
         sio.savemat(os.path.join(config.results_path, 'parameters.mat'), dictionary)
         
      return(result)

    
  def model(self):
    with tf.compat.v1.variable_scope("I_branch_F") as scope1:
      conv1_f_b,w1_f,bias1_f = conv2d(self.I_add, 1,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1_f = tf.nn.relu(conv1_f_b)
      pool1_f=max_pool_2x2(conv1_f)
      conv3_f_b,w3_f, bias3_f = conv2d(pool1_f, 64,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3_f = tf.nn.relu(conv3_f_b)
      pool2_f=max_pool_2x2(conv3_f)
      conv5_f_b,w5_f, bias5_f = conv2d(pool2_f, 128,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5_f = tf.nn.relu(conv5_f_b)
      pool3_f=max_pool_2x2(conv5_f)
      conv7_f_b, w7_f, bias7_f = conv2d(pool3_f,  1024, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7_f = tf.nn.relu(conv7_f_b)
      
      
    with tf.compat.v1.variable_scope("main_branch") as scope3:
      two_depths_input = tf.concat(axis = 3, values = [self.images, self.images_2]) 
      
      conv1_b, w1, bias1 = conv2d(two_depths_input, 2,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1 = tf.nn.relu(conv1_b)
      conv2_b, w2, bias2 = conv2d(conv1, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2")
      conv2 = tf.nn.relu(conv2_b)
      pool1=max_pool_2x2(conv2)
      
      pool1_input = self.pool1_input
      conv_input1_b, w_input1, bias_input1 = conv2d(pool1_input, 1,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input1")
      conv_input1 = tf.nn.relu(conv_input1_b)
      concate_input1=tf.concat(axis = 3, values = [pool1, conv_input1])  

      conv3_b, w3, bias3 = conv2d(concate_input1, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3 = tf.nn.relu(conv3_b)
      
      conv4_b, w4, bias4 = conv2d(conv3, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4")
      conv4 = tf.nn.relu(conv4_b)
      pool2=max_pool_2x2(conv4)

      pool2_input = self.pool2_input 
      conv_input2_b, w_input2, bias_input2 = conv2d(pool2_input, 1,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input2")
      conv_input2 = tf.nn.relu(conv_input2_b)
      concate_input2=tf.concat(axis = 3, values = [pool2,conv_input2])  

      conv5_b, w5, bias5 = conv2d(concate_input2, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5 = tf.nn.relu(conv5_b)
      conv6_b, w6, bias6 = conv2d(conv5, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6")
      conv6 = tf.nn.relu(conv6_b)
      pool3 = max_pool_2x2(conv6)

      pool3_input = self.pool3_input 
      conv_input3_b, w_input3, bias_input3 = conv2d(pool3_input, 1,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input3")
      conv_input3 = tf.nn.relu(conv_input3_b)
      concate_input3=tf.concat(axis = 3, values = [pool3,conv_input3])        

      conv7_b, w7, bias7 = conv2d(concate_input3, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7 = tf.nn.relu(conv7_b)
      conv8_b, w8, bias8 = conv2d(conv7, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8")
      conv8 = tf.nn.relu(conv8_b)
      pool4=max_pool_2x2(conv8)
     
      pool4_input = self.pool4_input 
      conv_input4_b, w_input4, bias_input4 = conv2d(pool4_input, 1,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input4")
      conv_input4 = tf.nn.relu(conv_input4_b)
      concate_input4=tf.concat(axis = 3, values = [pool4,conv_input4])        

      conv9_b, w9, bias9 = conv2d(concate_input4, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9")
      conv9 = tf.nn.relu(conv9_b)
      conv10_b, w10, bias10 = conv2d(conv9, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10")
      conv10 = tf.nn.relu(conv10_b)
      

      deconv2_b, w_deconv2, bias_deconv2 = deconv2d(conv10, conv8.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_2")
      deconv2 = tf.nn.relu(deconv2_b) 
      conb2 = tf.concat(axis = 3, values = [deconv2,conv8,conv7_f])  
      conv13_b, w13, bias13 = conv2d(conb2, 3072,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13")
      conv13 = tf.nn.relu(conv13_b)
      conv14_b, w14, bias14 = conv2d(conv13, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_14")
      conv14 = tf.nn.relu(conv14_b)
      deconv3_b, w_deconv3, bias_deconv3 = deconv2d(conv14, conv6.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_3")
      deconv3 = tf.nn.relu(deconv3_b)                          
      conb3 = tf.concat(axis = 3, values = [deconv3,conv6,conv5_f])
      conv15_b, w15, bias15 = conv2d(conb3, 768,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_15")
      conv15 = tf.nn.relu(conv15_b)      
      conv16_b, w16, bias16 = conv2d(conv15, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_16")
      conv16 = tf.nn.relu(conv16_b)
      deconv4_b, w_deconv4, bias_deconv4 = deconv2d(conv16, conv4.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_4")
      deconv4 = tf.nn.relu(deconv4_b)    
      conb4 = tf.concat(axis = 3, values = [deconv4,conv4,conv3_f])
      conv17_b, w17, bias17 = conv2d(conb4, 384,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_17")
      conv17 = tf.nn.relu(conv17_b)      
      conv18_b, w18, bias18 = conv2d(conv17, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_18")
      conv18 = tf.nn.relu(conv18_b) 
      deconv5_b, w_deconv5, bias_deconv5 = deconv2d(conv18, conv2.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_5")
      deconv5 = tf.nn.relu(deconv5_b)
      conb5 = tf.concat(axis = 3, values = [deconv5,conv2,conv1_f])
      conv19_b, w19, bias19 = conv2d(conb5, 192,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_19")
      conv19 = tf.nn.relu(conv19_b)      
      conv20_b, w20, bias20 = conv2d(conv19, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_20")
      conv20 = tf.nn.relu(conv20_b) 
      residual_map, w_output, bias_output = conv2d(conv20, 64,1, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_21")
      output = tf.add(residual_map,self.images) 

      dictionary = {'data':{'labels':self.labels, 'depth_down':self.images, 'depth_down_2':self.images_2, 'I_add':self.I_add},\
        'results':{'result' : output, 'residual_map':residual_map},\
          'features':{'pool1_input':pool1_input, 'pool2_input':pool2_input, 'pool3_input':pool3_input, 'pool4_input':pool4_input},\
            'features_intensity':{'pool1_f':pool1_f,'pool2_f':pool2_f, 'pool3_f':pool3_f },\
            '1st_conv':{'w1':w1, 'conv1_b':conv1_b}}


    return output, dictionary, residual_map

  def model_test(self):
    with tf.compat.v1.variable_scope("I_branch_F", reuse = True):
      conv1_f,w1_f,bias1_f = conv2d(self.I_add_test, 1,64,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1_f = tf.nn.relu(conv1_f)
      pool1_f=max_pool_2x2(conv1_f)
      conv3_f,w3_f, bias3_f = conv2d(pool1_f, 64,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3_f = tf.nn.relu(conv3_f)
      pool2_f=max_pool_2x2(conv3_f)
      conv5_f,w5_f, bias5_f = conv2d(pool2_f, 128,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5_f = tf.nn.relu(conv5_f)
      pool3_f=max_pool_2x2(conv5_f)
      conv7_f,w7_f, bias7_f = conv2d(pool3_f, 1024,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7_f = tf.nn.relu(conv7_f)

    with tf.compat.v1.variable_scope("main_branch", reuse = True):
      two_depths_input = tf.concat(axis = 3, values = [self.depth_test, self.depth_test_2]) 
      conv1, w1, bias1 = conv2d(two_depths_input, 2,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_1")
      conv1 = tf.nn.relu(conv1)
      conv2, w2, bias2 = conv2d(conv1, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_2")
      conv2 = tf.nn.relu(conv2)
      pool1=max_pool_2x2(conv2)
      
      pool1_input=self.pool1_input_test 
      conv_input1, w_input1, bias_input1 = conv2d(pool1_input, 1,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input1")
      conv_input1 = tf.nn.relu(conv_input1)
      concate_input1=tf.concat(axis = 3, values = [pool1,conv_input1])  

      conv3, w3, bias3 = conv2d(concate_input1, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_3")
      conv3 = tf.nn.relu(conv3)
      conv4, w4, bias4 = conv2d(conv3, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_4")
      conv4 = tf.nn.relu(conv4)
      pool2=max_pool_2x2(conv4)

      pool2_input = self.pool2_input_test
      conv_input2, w_input2, bias_input2 = conv2d(pool2_input, 1,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input2")
      conv_input2 = tf.nn.relu(conv_input2)
      concate_input2=tf.concat(axis = 3, values = [pool2,conv_input2])  

      conv5, w5, bias5 = conv2d(concate_input2, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_5")
      conv5 = tf.nn.relu(conv5)
      conv6, w6, bias6  = conv2d(conv5, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_6")
      conv6 = tf.nn.relu(conv6)
      pool3=max_pool_2x2(conv6)

      pool3_input = self.pool3_input_test
      conv_input3, w_input3, bias_input3 = conv2d(pool3_input, 1,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input3")
      conv_input3 = tf.nn.relu(conv_input3)
      concate_input3=tf.concat(axis = 3, values = [pool3,conv_input3])        

      conv7, w7, bias7 = conv2d(concate_input3, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_7")
      conv7 = tf.nn.relu(conv7)
      conv8, w8, bias8 = conv2d(conv7, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_8")
      conv8 = tf.nn.relu(conv8)
      pool4=max_pool_2x2(conv8)
     
      pool4_input=self.pool4_input_test
      conv_input4, w_input4, bias_input4 = conv2d(pool4_input, 1,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_input4")
      conv_input4 = tf.nn.relu(conv_input4)
      concate_input4=tf.concat(axis = 3, values = [pool4,conv_input4])        

      conv9, w9, bias9 = conv2d(concate_input4, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_9")
      conv9 = tf.nn.relu(conv9)
      conv10, w10, bias10 = conv2d(conv9, 1024,1024, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_10")
      conv10 = tf.nn.relu(conv10)
      
      deconv2, w_deconv2, bias_deconv2 = deconv2d(conv10, conv8.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_2")
      deconv2 = tf.nn.relu(deconv2) 
      conb2 = tf.concat(axis = 3, values = [deconv2,conv8,conv7_f])  
      conv13, w13, bias13 = conv2d(conb2, 3072,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_13")
      conv13 = tf.nn.relu(conv13)
      conv14, w14, bias14 = conv2d(conv13, 512,512, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_14")
      conv14 = tf.nn.relu(conv14)
      deconv3, w_deconv3, bias_deconv3 = deconv2d(conv14, conv6.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_3")
      deconv3 = tf.nn.relu(deconv3)                          
      conb3 = tf.concat(axis = 3, values = [deconv3,conv6,conv5_f])
      conv15, w15, bias15 = conv2d(conb3, 768,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_15")
      conv15 = tf.nn.relu(conv15)      
      conv16, w16, bias16 = conv2d(conv15, 256,256, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_16")
      conv16 = tf.nn.relu(conv16)
      deconv4, w_deconv4, bias_deconv4 = deconv2d(conv16, conv4.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_4")
      deconv4 = tf.nn.relu(deconv4)    
      conb4 = tf.concat(axis = 3, values = [deconv4,conv4,conv3_f])
      conv17, w17, bias17 = conv2d(conb4, 384,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_17")
      conv17 = tf.nn.relu(conv17)      
      conv18, w18, bias18 = conv2d(conv17, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_18")
      conv18 = tf.nn.relu(conv18) 
      deconv5, w_deconv5, bias_deconv5 = deconv2d(conv18, conv2.get_shape().as_list(), k_h=3, k_w=3, d_h=2, d_w=2,name="deconv2d_5")
      deconv5 = tf.nn.relu(deconv5)
      conb5 = tf.concat(axis = 3, values = [deconv5,conv2,conv1_f])
      conv19, w19, bias19 = conv2d(conb5, 192,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_19")
      conv19 = tf.nn.relu(conv19)      
      conv20, w20, bias20 = conv2d(conv19, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_20")
      conv20 = tf.nn.relu(conv20) 
      residual_map, w_output, bias_output = conv2d(conv20, 64,1, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_21")
      output = tf.add(residual_map,self.depth_test) 

      dictionary = {'data':{'depth_down':self.depth_test, 'depth_down_2':self.depth_test_2, 'I_add':self.I_add_test},\
        'results':{'result' : output, 'residual_map':residual_map},\
          'features':{'pool1_input':pool1_input, 'pool2_input':pool2_input, 'pool3_input':pool3_input, 'pool4_input':pool4_input},\
            'features_intensity':{'pool1_f':pool1_f,'pool2_f':pool2_f, 'pool3_f':pool3_f},\
            '1st_conv':{'w1':w1, 'conv1_b': conv1}}

    return output, dictionary

  
  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    #print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    #print("model_dir="+str(model_dir))
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    #print("checkpoint_dir"+str(checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #print("ckpt"+str(ckpt))
    #print("ckpt.model_checkpoint_path"+str(ckpt.model_checkpoint_path))
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("CHECKPOINT DETAILS : ckpt_name"+str(ckpt_name))
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
