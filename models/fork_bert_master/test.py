#coding=utf-8


"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""

import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensorflow.contrib import slim

from tensorflow.contrib.learn import ModeKeys

from tensorflow.contrib.learn import learn_runner


