#coding: utf-8

import tensorflow as tf

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

def sample_latents(means, logvars):
    # reparameterize
    noises = tf.random.normal(tf.shape(means))
    latents = means + tf.exp(0.5 * logvars) * noises
    return latents

def compute_kl_loss(means, logvars, means_prior=None, logvars_prior=None):
    if means_prior is None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means) - tf.exp(logvars) + 1.0), 1) # sum over latent dimentsion    
    elif means_prior is not None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means-means_prior) - tf.exp(logvars) + 1.0), 1) # sum over latent dimentsion    
    return kl_losses

def softmax_with_temperature(logits, axis=None, name=None, temperature=1.):
    if axis is None:
        axis = -1
    return tf.exp(logits / temperature) / tf.reduce_sum(tf.exp(logits / temperature), axis=axis)