#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:52:23 2020

@author: ashleyspindler
"""

from keras.layers import Layer
from keras import backend as K
from keras import initializers, constraints, regularizers
import numpy as np

class SoftMax(constraints.Constraint):
  """Returns SoftMax of Weight Tensor so that
         sum(W) = scale+offset"""

  def __init__(self, scale=1, offset=0, axis=0):
    self.scale = scale
    self.offset = offset
    self.axis = axis

  def __call__(self, w):
    soft = K.softmax(w, axis=self.axis)
    return soft*self.scale + self.offset

  def get_config(self):
    return {'scale': self.scale,
            'offset' : self.offset,
            'axis' : self.axis}
    
class SumNorm(constraints.Constraint):
  """Returns normalised weight tensor
         Wn = (w-min(w))/(max(w)-min(w))
         Wout = (w/sum(w)*scale+offset"""

  def __init__(self, scale=1, offset=0, axis=0):
    self.scale = scale
    self.offset = offset
    self.axis = axis

  def __call__(self, w):
    clipped = K.clip(w, K.abs(K.epsilon()), 1)
    scaled = clipped/K.sum(clipped, axis=self.axis)
    return scaled#*self.scale + self.offset

  def get_config(self):
    return {'scale': self.scale,
            'offset' : self.offset,
            'axis' : self.axis}
    
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
        
class GMMLayer(Layer):

    """
      Holds trainable weights for GMM properties, adds KL loss 
      between sampled Z and the Mixture of Gaussians
    """

    def __init__(self, latent_dims, n_centroids, *args, **kwargs):
        self.latent_dims = latent_dims
        self.n_centroids = n_centroids
        super(GMMLayer, self).__init__(*args, **kwargs)
        
    def get_config(self):
        config = {'latent_dims': self.latent_dims,
                  'n_centroids': self.n_centroids}
        base_config = super(GMMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.theta_p = self.add_weight(name='theta_p',
                                       shape=(self.n_centroids,),
                                       initializer=initializers.Constant(1/self.n_centroids),
                                       #initializer=initializers.RandomUniform(1/self.n_centroids,0.1),
                                       trainable=True,
                                       #constraint=constraints.NonNeg(),#SumNorm(1, axis=0),
                                       )#regularizer=regularizers.l1(0.001))
        self.u_p = self.add_weight(name='u_p',
                                   shape=(self.n_centroids,self.latent_dims),
                                   #initializer=initializers.he_uniform(),
                                   initializer=initializers.RandomUniform(-2,2),
                                   trainable=True,
                                   )#regularizer=regularizers.l2(0.01))
        self.lambda_p = self.add_weight(name='lambda_p',
                                        shape=(self.n_centroids,self.latent_dims),
                                        initializer=initializers.Ones(),
                                        trainable=True,
                                        constraint=constraints.NonNeg(),
                                        )#regularizer=regularizers.l2(0.01))
        super(GMMLayer, self).build(input_shape)

    def call(self, inputs):

        z = inputs

        theta_ = K.softmax(self.theta_p)
        lambda_ = K.log((self.lambda_p+1e-6))

        theta_t = (K.expand_dims(theta_,axis=-1))*K.ones((self.n_centroids,self.latent_dims))

        p_c_z = K.exp(K.sum(K.log(theta_t) - 0.5 * K.log(2 * np.pi) * lambda_ -
                            K.square(K.repeat(z,self.n_centroids) - self.u_p) /
                            (2 * K.exp(lambda_)), axis=-1))+1e-10

        gamma = (p_c_z/K.sum(p_c_z, axis=-1, keepdims=True))

        return [z, gamma, theta_, self.u_p, lambda_]

    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        return [(None,self.latent_dims), (None,self.n_centroids,), (self.n_centroids,), (self.n_centroids,self.latent_dims), (self.n_centroids,self.latent_dims)]

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
                                       #initializer=initializers.RandomUniform(1/self.n_centroids,0.1),
                                       trainable=True,
                                       #constraint=NonZero(),
                                       #regularizer=regularizers.l1(0.001)
                                       )
        self.u_p = self.add_weight(name='u_p',
                                   shape=(self.latent_dims,self.n_centroids),
                                   #initializer=initializers.he_uniform(),
                                   initializer=initializers.RandomUniform(-2,2),
                                   trainable=True,
                                   #regularizer=regularizers.l2(0.01)
                                   )
        self.lambda_p = self.add_weight(name='lambda_p',
                                        shape=(self.latent_dims,self.n_centroids),
                                        initializer=initializers.Ones(),
                                        trainable=True,
                                        constraint=NonZero(),
                                        #regularizer=regularizers.l2(0.01)
                                        )
        super(GMMLayer_2, self).build(input_shape)

    def call(self, inputs):

        z = inputs

        return z

    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        return (None,self.latent_dims)
