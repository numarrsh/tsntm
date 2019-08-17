#coding: utf-8
import numpy as np
import tensorflow as tf
from components import tf_log, sample_latents, compute_kl_loss

class TopicModel():
    def __init__(self, config, bow_idxs):
        self.config = config
        self.bow_idxs = bow_idxs
        
        t_variables = {}
        t_variables['bow'] = tf.placeholder(tf.float32, [None, self.config.dim_bow])
        t_variables['keep_prob'] = tf.placeholder(tf.float32)
        self.t_variables = t_variables
        
    def build(self):
        # encode bow
        with tf.variable_scope('topic/enc', reuse=False):
            hidden_bow_ = tf.keras.layers.Dense(units=self.config.dim_hidden_bow, activation=tf.nn.relu, name='hidden')(self.t_variables['bow'])
            hidden_bow = tf.keras.layers.Dropout(self.t_variables['keep_prob'])(hidden_bow_)
            means_bow = tf.keras.layers.Dense(units=self.config.dim_latent_topic)(hidden_bow)
            logvars_bow = tf.keras.layers.Dense(units=self.config.dim_latent_topic, kernel_initializer=tf.constant_initializer(0), bias_initializer=tf.constant_initializer(0))(hidden_bow)
            latents_bow = sample_latents(means_bow, logvars_bow) # sample latent vectors

            prob_topic = tf.layers.Dense(units=self.config.n_topic, activation=tf.nn.softmax, name='prob')(latents_bow) # inference of topic probabilities

        # decode bow
        with tf.variable_scope('shared', reuse=False):
            embeddings = tf.get_variable('emb', [self.config.n_vocab, self.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of vocab

        bow_embeddings = tf.nn.embedding_lookup(embeddings, self.bow_idxs) # embeddings of each bow features

        with tf.variable_scope('topic/dec', reuse=False):
            topic_embeddings = tf.get_variable('topic_emb', [self.config.n_topic, self.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of topics

            topic_bow = tf.nn.softmax(tf.matmul(topic_embeddings, bow_embeddings, transpose_b=True), 1) # bow vectors for each topic
            self.topic_bow = topic_bow
            prob_bow = tf_log(tf.matmul(prob_topic, topic_bow)) # predicted bow distribution

        # define lisses
        topic_losses_recon = -tf.reduce_sum(tf.multiply(self.t_variables['bow'], prob_bow), 1)
        topic_loss_recon = tf.reduce_mean(topic_losses_recon) # negative log likelihood of each words
        self.topic_losses_recon, self.topic_loss_recon = topic_losses_recon, topic_loss_recon

        topic_loss_kl = compute_kl_loss(means_bow, logvars_bow) # KL divergence b/w latent dist & gaussian std
        self.topic_loss_kl = topic_loss_kl

        topic_embeddings_norm = topic_embeddings / tf.norm(topic_embeddings, axis=1, keepdims=True)
        topic_angles = tf.matmul(topic_embeddings_norm, tf.transpose(topic_embeddings_norm))
        topic_angles_mean = tf.reduce_mean(topic_angles, keepdims=True)
        topic_angles_vars = tf.reduce_mean(tf.square(topic_angles - topic_angles_mean))
        topic_loss_reg = topic_angles_vars - tf.squeeze(topic_angles_mean)
        self.topic_loss_reg = topic_loss_reg

    def build_opt(self):        
        if self.config.warmup_topic > 0:
            beta = tf.Variable(self.config.beta, name='beta', trainable=False)
            update_beta = tf.assign_add(beta, 1./(self.config.warmup_topic*num_train_batches))
            self.beta, self.update_beta = beta, update_beta
            loss = self.topic_loss_recon + self.beta * self.topic_loss_kl + self.config.reg * self.topic_loss_reg
        else:
            loss = self.topic_loss_recon + self.topic_loss_kl + self.config.reg * self.topic_loss_reg
        self.loss = loss
        
        # define optimizer
        if self.config.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif self.config.opt == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(self.config.lr)

        grad_vars = optimizer.compute_gradients(loss)
        clipped_grad_vars = [(tf.clip_by_value(grad, -self.config.grad_clip, self.config.grad_clip), var) for grad, var in grad_vars]
        opt = optimizer.apply_gradients(clipped_grad_vars)
        self.opt = opt

        n_bow = tf.reduce_sum(self.t_variables['bow'], 1)
        ppls = tf.divide(self.topic_losses_recon, n_bow)
        self.ppls = ppls
        
        topics_freq_bow_indices = tf.nn.top_k(self.topic_bow, 10, name='topic_freq_bow').indices
        self.topics_freq_bow_indices = topics_freq_bow_indices
        
    def get_feed_dict(self, batch, mode='train'):
        bow = np.array([instance.bow for instance in batch]).astype(np.float32)
        keep_prob = self.config.keep_prob if mode == 'train' else 1.0
        feed_dict = {
                    self.t_variables['bow']: bow, 
                    self.t_variables['keep_prob']: keep_prob
        }
        return  feed_dict