#coding: utf-8

from collections import defaultdict

import copy
import numpy as np
import tensorflow as tf

from components import tf_log, sample_latents, compute_kl_loss, softmax_with_temperature

class GaussianSoftmaxModel():
    def __init__(self, config):
        self.config = config
        
        self.t_variables = {}
        self.build()
        
    def build(self):
        def get_topic_loss_reg(topic_embeddings):
            topic_embeddings_norm = topic_embeddings / tf.norm(topic_embeddings, axis=1, keepdims=True)
            self.topic_dots = tf.clip_by_value(tf.matmul(topic_embeddings_norm, tf.transpose(topic_embeddings_norm)), -1., 1.)
            topic_loss_reg = tf.reduce_mean(tf.square(self.topic_dots - tf.eye(self.config.n_topic)))
            return topic_loss_reg
           
        # -------------- Build Model --------------
        tf.reset_default_graph()
        
        tf.set_random_seed(self.config.seed)
        
        self.t_variables['bow'] = tf.placeholder(tf.float32, [None, self.config.dim_bow])
        self.t_variables['keep_prob'] = tf.placeholder(tf.float32)
        
        # encode bow
        with tf.variable_scope('topic/enc', reuse=False):
            hidden_bow_ = tf.layers.Dense(units=self.config.dim_hidden_bow, activation=tf.nn.tanh, name='hidden_bow')(self.t_variables['bow'])
            hidden_bow = tf.layers.Dropout(self.t_variables['keep_prob'])(hidden_bow_)
            means_bow = tf.layers.Dense(units=self.config.dim_latent_bow, name='mean_bow')(hidden_bow)
            logvars_bow = tf.layers.Dense(units=self.config.dim_latent_bow, kernel_initializer=tf.constant_initializer(0), bias_initializer=tf.constant_initializer(0), name='logvar_bow')(hidden_bow)
            latents_bow = sample_latents(means_bow, logvars_bow) # sample latent vectors
            
            self.prob_topic = tf.layers.Dense(units=self.config.n_topic, activation=tf.nn.softmax, name='prob_topic')(latents_bow) # inference of topic probabilities

        # decode bow
        with tf.variable_scope('shared', reuse=False):
            self.bow_embeddings = tf.get_variable('emb', [self.config.dim_bow, self.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of vocab

        with tf.variable_scope('topic/dec', reuse=False):
            emb_layer = lambda h: tf.layers.Dense(units=self.config.dim_emb, name='output')(tf.nn.tanh(h))
            self.topic_embeddings = tf.get_variable('topic_emb', [self.config.n_topic, self.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of topics

            self.topic_bow = tf.nn.softmax(tf.matmul(self.topic_embeddings, self.bow_embeddings, transpose_b=True), 1) # bow vectors for each topic
            self.logits_bow = tf_log(tf.matmul(self.prob_topic, self.topic_bow)) # predicted bow distribution N_Batch x  V
            
        # define losses
        self.topic_losses_recon = -tf.reduce_sum(tf.multiply(self.t_variables['bow'], self.logits_bow), 1)
        self.topic_loss_recon = tf.reduce_mean(self.topic_losses_recon) # negative log likelihood of each words
        self.topic_loss_kl = compute_kl_loss(means_bow, logvars_bow) # KL divergence b/w latent dist & gaussian std
        self.topic_loss_reg = get_topic_loss_reg(self.topic_embeddings)
        self.loss = self.topic_loss_recon + self.topic_loss_kl + self.config.reg * self.topic_loss_reg

        # define optimizer
        if self.config.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif self.config.opt == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(self.config.lr)

        self.grad_vars = optimizer.compute_gradients(self.loss)
        self.clipped_grad_vars = [(tf.clip_by_value(grad, -self.config.grad_clip, self.config.grad_clip), var) for grad, var in self.grad_vars]
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.opt = optimizer.apply_gradients(self.clipped_grad_vars, global_step=self.global_step)

        # monitor
        self.n_bow = tf.reduce_sum(self.t_variables['bow'], 1)
        self.topic_ppls = tf.divide(self.topic_losses_recon, tf.maximum(1e-5, self.n_bow))
    
        # growth criteria
        self.n_topics = tf.multiply(tf.expand_dims(self.n_bow, -1), self.prob_topic)
    
    def get_feed_dict(self, batch, mode='train'):
        bow = np.array([instance.bow for instance in batch]).astype(np.float32)
        keep_prob = self.config.keep_prob if mode == 'train' else 1.0
        feed_dict = {
                    self.t_variables['bow']: bow, 
                    self.t_variables['keep_prob']: keep_prob
        }
        return  feed_dict