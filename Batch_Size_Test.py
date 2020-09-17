#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:40:42 2020

@author: ashleyspindler

script to test running speed over different batch sizes
"""

from keras import models

import configparser
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras import optimizers
from keras.utils import multi_gpu_model
from skimage.exposure import rescale_intensity
import time

#%%============================================================================
# Functions

# Losses
def KL_Loss(z_mean, z_log_var, original_dims, loss_type, s=1):
    def vae_loss(y_true, y_pred):

        if loss_type=='binary_crossentropy':
            recon_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*original_dims
        else:
            recon_loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*original_dims

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        tot_loss = K.mean(recon_loss + s*kl_loss, axis=-1)

        return tot_loss
    return vae_loss

def KL_metric(z_mean, z_log_var):
    def kl_loss(y_true, y_pred):

        loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        return K.mean(loss, axis=-1)
    return kl_loss

def recon_metric(loss_type, original_dims):
    def recon_loss(y_true, y_pred):

        if loss_type=='binary_crossentropy':
            loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*original_dims
        else:
            loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*original_dims

        return loss
    return recon_loss

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

def contrast_enhance(img):

    img_rescale = rescale_intensity(img, in_range=(30, 255), out_range=(0,255))
    return img_rescale

#%%============================================================================

config = configparser.ConfigParser()
config.read('ae-config.txt')

Model_Name = config['ae_settings']['Model_Name']
side = int(config['ae_settings']['side'])
b = int(config['ae_settings']['b'])
cm = 'grayscale' if b==1 else 'rgb'
batch_size = [16, 32, 64, 128, 256]
n_train = int(config['ae_settings']['n_train'])


train_dir = '/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Images/Train'

#%%============================================================================

T = []

for BS in batch_size:
    print('Testing Batch Size:', BS)
    steps = n_train//BS
    datagen = ImageDataGenerator(rescale=1./255,
                             preprocessing_function=contrast_enhance)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(side,side),
                                              color_mode=cm, class_mode='input',
                                              batch_size=BS)
    
    loss_type = 'binary_crossentropy'
    customs = {'recon_loss' : recon_metric(loss_type, 128*128*3),
           'vae_loss' : recon_metric(loss_type, 128*128*3),
           'kl_loss' : recon_metric(loss_type, 128*128*3)}

    autoencoder = models.load_model('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/untrained_astrovader'+Model_Name+'.h5',
                                    custom_objects=customs)    
    z_mean = autoencoder.get_layer('latentmean').output
    z_log_var = autoencoder.get_layer('latentlog_var').output
    
    losses = KL_Loss(z_mean, z_log_var, side*side*b, loss_type, 1) # VCAE
    metrics = [recon_metric(loss_type, side*side*b), KL_metric(z_mean, z_log_var)] # VCAE

    ae_par = multi_gpu_model(autoencoder, gpus=2)
    ae_par.compile(optimizers.Adam(lr=1e-3),
                   loss = losses,
                   metrics = metrics)
    start = time.time()
    ae_par.fit_generator(train_generator, epochs=10,
                          steps_per_epoch=steps, initial_epoch=0,
                          workers=20, use_multiprocessing=True,
                          verbose=1,
                          )
    stop = time.time()
    print('Batch size', BS, 'trained 10 Epochs in:', stop-start, 'seconds')
    T.append(stop-start)
    K.clear_session()
    
print(T)


