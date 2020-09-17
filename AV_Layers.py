#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:29:30 2020

@author: ashleyspindler
Lambda Layer Functions for AstroVader
"""

import keras.backend as K
from keras.layers import Layer
from keras import initializers, constraints

class NonZero(constraints.Constraint):
    """Constrains the weights to be non-negative.
    casts tensor with locations of w<=0
    computes relu of weights
    add 1e-7 to zero value weights
    """

    def __call__(self, w):
        non_positive = K.cast(K.less_equal(w,0),dtype='float32')
        w_relu = K.relu(w)
        return w_relu + non_positive * K.abs(K.epsilon())


def VAE_sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class get_gamma(Layer):
    
    def __init__(self, weights, params, *args, **kwargs):
        self.theta = weights['theta']
        self.mu = weights['mu']
        self.sigma = weights['lambda']        
        self.latents = params['latents']
        self.clusters = params['clusters']
        super(get_gamma, self).__init__(*args, **kwargs)
    
    def get_config(self):
        config = {'theta': self.theta,
                  'mu' : self.mu,
                  'sigma' : self.sigma,
                  'latents' : self.latents,
                  'clusters' : self.clusters}
        base_config = super(get_gamma, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        """
        Calculate ln p(c|z) for given z_mean, using the inverse min-max scaling 
        from https://arxiv.org/pdf/2005.08047.pdf
        """
        
        z_mu = inputs
        log2pi = 1.8378770664093453
        
        precisions = 1.0 / self.sigma
        precisions_chol = 1.0 / K.sqrt(self.sigma)
        _log_prob = K.sum((K.square(self.mu) * precisions), axis=0)
        _log_prob -= 2.0 * K.dot(z_mu, self.mu * precisions)
        _log_prob += K.dot(K.square(z_mu), precisions)
        log_det_chol = K.sum(K.log(precisions_chol), axis=0)
        log_prob = -0.5 * (self.latents * log2pi + _log_prob) + log_det_chol
        log_weights = K.log(self.theta)
        
        _p_cz = -(log_prob + log_weights)
        
        _p_cz = (_p_cz - K.min(_p_cz))\
              / ((K.max(_p_cz) - K.min(_p_cz) + 1e-12))\
              * (-50.0)
        
        gamma = K.softmax(_p_cz, axis=-1)
    
        return [gamma, z_mu]
    
    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        return [(None,self.clusters), (None,self.latents)]

def cond_z(y_training):
    def _cond_z(args):
        return K.switch(y_training, args[0], args[1])
    return _cond_z

class GMMLayer_2(Layer):

    """
      Holds trainable weights for GMM properties, adds KL loss 
      between sampled Z and the Mixture of Gaussians
      
      Only returns z, acts as pass through layer for GMM weights
    """

    def __init__(self, latent_dims, n_centroids, *args, **kwargs):
        self.latent_dims = latent_dims
        self.n_centroids = n_centroids
        super(GMMLayer_2, self).__init__(*args, **kwargs)
        
    def get_config(self):
        config = {'latent_dims': self.latent_dims,
                  'n_centroids': self.n_centroids}
        base_config = super(GMMLayer_2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.theta_p = self.add_weight(name='theta_p',
                                       shape=(self.n_centroids,),
                                       initializer=initializers.Constant(1/self.n_centroids),
                                       trainable=True,
                                       )
        self.u_p = self.add_weight(name='u_p',
                                   shape=(self.latent_dims,self.n_centroids),
                                   initializer=initializers.he_uniform(),
                                   trainable=True,
                                   )
        self.lambda_p = self.add_weight(name='lambda_p',
                                        shape=(self.latent_dims,self.n_centroids),
                                        initializer=initializers.Ones(),
                                        trainable=True,
                                        constraint=NonZero(),
                                        )
        super(GMMLayer_2, self).build(input_shape)

    def call(self, inputs):

        z = inputs

        return z

    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        return (None,self.latent_dims)
