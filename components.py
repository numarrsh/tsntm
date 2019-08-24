#coding:utf-8

import tensorflow as tf
import numpy as np

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

# def encode_latents(x, dim, name):
#     with tf.variable_scope(name, reuse=False):
#         # encode to parameter 
#         means_logvars = tf.layers.Dense(units=dim*2, activation=tf.nn.relu)(x)
#         means, logvars = tf.split(means_logvars, 2, -1)
#     return means, logvars
        
def sample_latents(means, logvars):
    # reparameterize
    noises = tf.random.normal(tf.shape(means))
    latents = means + tf.exp(0.5 * logvars) * noises
    return latents

def compute_kl_loss(means, logvars, means_prior=None, logvars_prior=None):
    if means_prior is None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means) - tf.exp(logvars) + 1.0), 1) # sum over latent dimentsion    
        kl_loss = tf.reduce_mean(kl_losses, [0]) #mean of kl_losses over batches
    return kl_loss

def dynamic_rnn(inputs, seqlen, n_hidden, keep_prob, cell_name='', reuse=False):
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = keep_prob)
        fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name, reuse=reuse):
        outputs, output_state = tf.nn.dynamic_rnn(fw_cell, inputs,
                                                                 initial_state=fw_initial_state,
                                                                 sequence_length=seqlen, 
                                                                 dtype=tf.float32)
    return outputs, output_state    

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
        outputs, bi_output_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 sequence_length=seqlen)
    
    output_state = tf.concat(list(bi_output_state), 1)
    
    return outputs, output_state    

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