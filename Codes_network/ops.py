import math
import numpy as np 
import tensorflow as tf
import random
from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.compat.v1.summary.image
  scalar_summary = tf.compat.v1.summary.scalar
  histogram_summary = tf.compat.v1.summary.histogram
  merge_summary = tf.compat.v1.summary.merge
  SummaryWriter = tf.compat.v1.summary.FileWriter
except:
  image_summary = tf.compat.v1.summary.image
  scalar_summary = tf.compat.v1.summary.scalar
  histogram_summary = tf.compat.v1.summary.histogram
  merge_summary = tf.compat.v1.summary.merge
  SummaryWriter = tf.compat.v1.summary.FileWriter

# if "concat_v2" in dir(tf):
#   def concat(tensors, axis, *args, **kwargs):
#     return tf.concat_v2(tensors, axis, *args, **kwargs)
# else:
#   def concat(tensors, axis, *args, **kwargs):
#     return tf.concat(tensors, axis, *args, **kwargs)


def conv2d(input_, input_dim,output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.compat.v1.variable_scope(name):

    w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv,w, biases

    
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.compat.v1.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.compat.v1.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv, w, biases

def max_pool_2x2(x):
  return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


