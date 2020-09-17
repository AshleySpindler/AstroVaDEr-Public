#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:39:40 2020

@author: ashleyspindler
"""

from keras.layers import Input, Dense, Flatten, Reshape,\
                         MaxPooling2D, UpSampling2D, Conv2D, Lambda,\
                         GaussianNoise, LeakyReLU
import keras.backend as K
from keras import regularizers

from AV_Layers import VAE_sampling, get_gamma, cond_z, GMMLayer


def encoder_setup(In_Shape, filters, kernels, params):
    I = Input(shape=In_Shape, dtype='float32', name='encoder_input') # (?, 128, 128, b)
    Noise = GaussianNoise(1e-8)(I)
    
    Conv1 = Conv2D(filters[0], kernel_size=kernels[0],# activation='relu',
                   padding='same', name='Conv1',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Noise) # (?, 128, 128, ?)
    Conv1 = LeakyReLU(alpha=0.1)(Conv1)
    Conv2 = Conv2D(filters[0], kernel_size=kernels[0],# activation='relu',
                   padding='same', name='Conv2',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv1) # (?, 128, 128, ?)
    Conv2 = LeakyReLU(alpha=0.1)(Conv2)
    Pool1 = MaxPooling2D((2,2), name='Pool1')(Conv2) # (?, 64, 64, ?)
    
    Conv3 = Conv2D(filters[1], kernel_size=kernels[1],# activation='relu',
                   padding='same', name='Conv3',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Pool1) # (?, 64, 64, ?)
    Conv3 = LeakyReLU(alpha=0.1)(Conv3)
    Conv4 = Conv2D(filters[1], kernel_size=kernels[1],# activation='relu',
                   padding='same', name='Conv4',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv3) # (?, 64, 64, 32)
    Conv4 = LeakyReLU(alpha=0.1)(Conv4)
    Pool2 = MaxPooling2D((2,2), name='Pool2')(Conv4) # (?, 32, 32, ?)
    
    Conv5 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv5',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Pool2) # (?, 32, 32, ?)
    Conv5 = LeakyReLU(alpha=0.1)(Conv5)
    Conv6 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv6',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv5) # (?, 32, 32, ?)
    Conv6 = LeakyReLU(alpha=0.1)(Conv6)
    Pool3 = MaxPooling2D((2,2), name='Pool3')(Conv6) # (?, 16, 16, ?)
    
    
    Flat = Flatten(name='Flat')(Pool3) # (?, 16*16*?)

    return I, Flat

def embedding_setup(params, enc_out):
    z_mean = Dense(params['latents'], name='latentmean',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(enc_out) # Edit namespace for CAE
    z_log_var = Dense(params['latents'], name='latentlog_var',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(enc_out) # Comment out for CAE
    
    z = Lambda(VAE_sampling, output_shape=(params['latents'],),
               name='latentz_sampling')([z_mean,z_log_var]) # repara trick
    
    y_training = K.variable(False, dtype='bool')
    z_out = Lambda(cond_z(y_training), output_shape=(params['latents'],),
                   name='conditional_z_out')([z_mean, z]) # set input for decoder
    
    GMM = GMMLayer(params['latents'], params['clusters'], name='latentGMM_Layer')
    z_out = GMM(z_out) # pass through layer containing GMM weights
    
    gmm_weights = { 'theta' : GMM.weights[0],
                    'mu' : GMM.weights[1],
                    'lambda' : GMM.weights[2],}
    gamma_out, z_out = get_gamma(gmm_weights, params)(z_out)
    
    outputs = { 'z' : z,
                'z_mean' : z_mean,
                'z_log_var' : z_log_var,
                'z_out' : z_out,
                'gamma' : gamma_out,}
    
    return outputs, gmm_weights

def decoder_setup(flat_units, filters, kernels, params, dec_in=None):
    
    if dec_in is None:
        dec_in = Input(shape=(params['latents'],), dtype='float32', name='decoder_input')

    FC3 = Dense(flat_units[0], name='FC3',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(dec_in) # (?, 16*16*?)
    FC3 = LeakyReLU(alpha=0.1)(FC3)
    reshape = Reshape((16,16,filters[2]), name='reshape')(FC3) # (?, 16, 16, ?)
    
    Conv7 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv7',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(reshape) # (?, 16, 16, ?) 
    Conv7 = LeakyReLU(alpha=0.1)(Conv7)
    Conv8 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv8',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv7) # (?, 16, 16, ?) 
    Conv8 = LeakyReLU(alpha=0.1)(Conv8)
    Up1 = UpSampling2D((2,2), interpolation='bilinear',
                       name='Up1')(Conv8) # (?, 32, 32, ?)
    
    Conv9 = Conv2D(filters[1], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv9',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Up1) # (?, 32, 32, ?) 
    Conv9 = LeakyReLU(alpha=0.1)(Conv9)
    Conv10 = Conv2D(filters[1], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv10',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv9) # (?, 32, 32, ?) 
    Conv10 = LeakyReLU(alpha=0.1)(Conv10)
    Up2 = UpSampling2D((2,2), interpolation='bilinear',
                       name='Up2')(Conv10) # (?, 64, 64, ?)
    
    Conv11 = Conv2D(filters[0], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv11',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Up2) # (?, 64, 64, ?) 
    Conv11 = LeakyReLU(alpha=0.1)(Conv11)
    Conv12 = Conv2D(filters[0], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv12',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv11) # (?, 64, 64, ?) 
    Conv12 = LeakyReLU(alpha=0.1)(Conv12)
    Up3 = UpSampling2D((2,2), interpolation='bilinear',
                       name='Up3')(Conv12) # (?, 128, 128, ?)
    
    Out = Conv2D(params['bands'], kernel_size=kernels[0], activation=params['output_activity'],
                 padding='same', name='Output',
                 kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                 bias_regularizer=regularizers.l2(params['l2_regularizer']),
                 )(Up3) # (?, 128, 128, b)
    
    return dec_in, Out
