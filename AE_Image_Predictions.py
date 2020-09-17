#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:02:58 2020

@author: ashleyspindler

Script to load model and validation data then compare reconstructed images
"""

from keras import models

import configparser
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from skimage.exposure import rescale_intensity
from AE_GMM_Layer import GMMLayer

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

        tot_loss = K.mean(recon_loss + s*K.expand_dims(K.expand_dims(kl_loss/original_dims, axis=-1), axis=-1))

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

def VaDE_metric(z_mean, z_log_var, gamma, theta_, u_, lambda_, n_centroids, latent_dims):
    def vade_metric(y_true, y_pred):
        gmm_loss = 0.5 * K.sum(gamma * K.sum( lambda_ +
                                              K.repeat(K.exp(z_log_var), n_centroids) / K.exp(lambda_) +
                                              K.square( K.repeat(z_mean, n_centroids) - u_) / K.exp(lambda_),
                                              axis=2), axis=1)\
                       - K.sum(gamma * K.log(theta_/gamma), axis=1)\
                       - 0.5 * K.sum(1 + z_log_var, axis=1)

        return K.mean(gmm_loss, axis=-1)
    return vade_metric

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
latents = int(config['ae_settings']['latents'])
clusters = int(config['ae_settings']['clusters'])

valid_dir = '/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Valid_imgs'

datagen = ImageDataGenerator(rescale=1./255,
                             preprocessing_function=contrast_enhance)
test_generator = datagen.flow_from_directory(valid_dir, target_size=(side,side),
                                              color_mode=cm, class_mode='sparse',
                                              batch_size=550)

x_test, y_valid = test_generator.next()

loss_type = 'binary_crossentropy'
customs = {'recon_loss' : recon_metric(loss_type, 128*128*3),
           'vae_loss' : recon_metric(loss_type, 128*128*3),
           'kl_loss' : recon_metric(loss_type, 128*128*3),
           'GMMLayer' : GMMLayer,
           'vade_metric' : recon_metric(loss_type, 128*128*3)}

autoencoder = models.load_model('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader'+Model_Name+'.h5',
                                custom_objects=customs)


#%%============================================================================

# plot model
plot_model(autoencoder, to_file='/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader'+Model_Name+'_plot.png')

# predict reconstructions with autoencoder
x_pred = autoencoder.predict(x_test, batch_size=50)

# Plot a single image and its reconstruction

ind = np.random.randint(low=0, high=549, size=1)[0]

f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(x_test[ind,:,:,0], origin='right')
ax2.imshow(x_pred[ind,:,:,0], origin='right')
#plt.savefig('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Figures/Single_Prediction'+Model_Name+'.png')
plt.show()

gals = np.random.choice(5000, 60, replace=False)

f, ax = plt.subplots(15,12)
plt.subplots_adjust(wspace=0, hspace=0)
k = 0
for i in range(0,5):
    for j in range(0,12):
        g = gals[k]
        ax[i*3,j].imshow(x_test[g,:,:,0], vmin=0, vmax=1, cmap='binary')
        ax[i*3+1,j].imshow(x_pred[g,:,:,0], vmin=0, vmax=1, cmap='binary')
        ax[i*3+2,j].imshow((x_test[g,:,:,0]-x_pred[g,:,:,0]), cmap='binary')
        k+=1

[[a.axis('off') for a in ax.ravel()]]    
plt.show()
plt.savefig('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Figures/Multiple_Predictions'+Model_Name+'.png')
