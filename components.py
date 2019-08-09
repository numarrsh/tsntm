#coding:utf-8

import tensorflow as tf
import numpy as np

def dynamic_rnn(inputs, seqlen, n_hidden, keep_prob, cell_name='', reuse=False):
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = keep_prob)
        fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name, reuse=reuse):
        outputs, output_states = tf.nn.dynamic_rnn(fw_cell, inputs,
                                                                 initial_state=fw_initial_state,
                                                                 sequence_length=seqlen, 
                                                                 dtype=tf.float32)
    return outputs, output_states    

def dynamic_bi_rnn(inputs, seqlen, n_hidden, keep_prob, cell_name='', reuse=False):
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = keep_prob)
        fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name + 'bw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = keep_prob)
        bw_initial_state = bw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name, reuse=reuse):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 sequence_length=seqlen)
    return outputs, output_states    

def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar))

class DiagonalGaussian(object):
    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)