#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:47:20 2020

@author: ashleyspindler
"""

import configparser #used to read config file
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D, Conv2D, Lambda
import keras.backend as K
from keras import Model, optimizers
from keras.callbacks import ModelCheckpoint

from skimage.exposure import rescale_intensity

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
# Initial Setup

""" Hyperparameters
    Epochs - number of epochs to run
    BatchSize - number of imgs to process in a batch
    n_train - size of training sample
    steps - n_train/BatchSize
    In_shape - shape of input image
    filters - number of filters in each conv layer
    kernel_size - size of the convolutional kernel
    hidden_units - size of FC layers
    latents - size of latent space
    clusters - number of clusters to fit
    lr - Learning Rate
"""

config = configparser.ConfigParser()
config.read('ae-config.txt')

Model_Name = '_Cheng_'+config['ae_settings']['Model_Name']
VAE = config['ae_settings']['VAE']=='True'
Epochs = int(config['ae_settings']['Epochs'])
batch_size = int(config['ae_settings']['batch_size'])
n_train = int(config['ae_settings']['n_train'])
steps = n_train//batch_size
side = int(config['ae_settings']['side'])
b = int(config['ae_settings']['b'])
cm = 'grayscale' if b==1 else 'rgb'
In_shape = (side,side,b) #different input size to Cheng but should still work
filters = [128,64,32,16,8] #from Cheng et al 2020
kernels = [(8,8), (7,7), (3,3)]
hidden_units = [64,32,32,64,128] #from Cheng et al 2020
latents = int(config['ae_settings']['latents'])
clusters = int(config['ae_settings']['clusters'])
lr = float(config['ae_settings']['lr'])

#%%============================================================================
# Create image data generator - rotates images and improves contrast

train_dir = '/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Images/Train'
#valid_dir = '/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Valid_imgs'

datagen = ImageDataGenerator(rescale=1./255,#rotation_range=90,
                             horizontal_flip=True, vertical_flip=True,
                             dtype='float32', fill_mode='wrap',
                             preprocessing_function=contrast_enhance)
train_generator = datagen.flow_from_directory(train_dir, target_size=(side,side),
                                              color_mode=cm, class_mode='input',
                                              batch_size=batch_size,
                                              shuffle=True)


"""
From Cheng et al 2019, testing following CAE layout with and without variational
latent space

ENCODER
In > Conv1 > Pool1 > Conv2 > Pool2 > Conv3 > Pool3 > Conv4 > Pool4 >
     Conv5 > Pool5 > Flat > FC1 > FC2 > Latent
     
DECODER
Latent > FC3 > FC4 > FC5 > Reshape > Conv6 > Up1 > Conv7 > Up2 > Conv8 >
         Up3 > Conv9 > Up4 > Conv10 > Up5 > Conv11 > Out

"""
#%%============================================================================
# Encoder Setup
I = Input(shape=In_shape, dtype='float32', name='input') # (?, 128, 128, b)

Conv1 = Conv2D(filters[0], kernel_size=kernels[0], activation='relu',
               padding='same', name='Conv1')(I) # (?, 128, 128, 128)
Pool1 = MaxPooling2D((2,2), name='Pool1')(Conv1) # (?, 64, 64, 128)

Conv2 = Conv2D(filters[1], kernel_size=kernels[1], activation='relu',
               padding='same', name='Conv2')(Pool1) # (?, 64, 64, 64)
Pool2 = MaxPooling2D((2,2), name='Pool2')(Conv2) # (?, 32, 32, 64)

Conv3 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv3')(Pool2) # (?, 32, 32, 32)
Pool3 = MaxPooling2D((2,2), name='Pool3')(Conv3) # (?, 16, 16, 32)

Conv4 = Conv2D(filters[3], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv4')(Pool3) # (?, 16, 16, 16)
Pool4 = MaxPooling2D((2,2), name='Pool4')(Conv4) # (?, 8, 8, 16)

Conv5 = Conv2D(filters[4], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv5')(Pool4) # (?, 8, 8, 8)
Pool5 = MaxPooling2D((2,2), name='Pool5')(Conv5) # (?, 4, 4, 8)

Flat = Flatten(name='Flat')(Pool5) # (?, 128)
FC1 = Dense(hidden_units[0], activation='relu', name='FC1')(Flat) # (?, 64)
FC2 = Dense(hidden_units[1], activation='relu', name='FC2')(FC1) # (?, 32)

"""Embedded Layer - Option for Variational or not"""
if VAE:
    z_mean = Dense(latents, name='latentmean')(FC2) # Edit namespace for CAE
    z_log_var = Dense(latents, name='latentlog_var')(FC2) # Comment out for CAE
    z = Lambda(VAE_sampling, output_shape=(latents,),
           name='latentz_sampling')([z_mean,z_log_var]) # Comment out for CAE
else:
    z = Dense(latents, name='latentmean')(FC2) # Edit namespace for CAE

#%%============================================================================
# Decoder Setup

FC3 = Dense(hidden_units[2], activation='relu', name='FC3')(z) # (?, 32)
FC4 = Dense(hidden_units[3], activation='relu', name='FC4')(FC3) # (?, 64)
FC5 = Dense(hidden_units[4], activation='relu', name='FC5')(FC4) # (?, 128)
Reshape = Reshape((4,4,8))(FC5) # (?, 4, 4, 8)

# Cheng uses Conv > Up, but I've found vice versa works better, need to test
# always seemed like the flatten/reshape are taking the place of dense layers
# in an AE

Conv6 = Conv2D(filters[4], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv6')(Reshape) # (?, 4, 4, 8)
Up1 = UpSampling2D((2,2), interpolation='nearest', name='Up1')(Conv6) # (?, 8, 8, 8)

Conv7 = Conv2D(filters[3], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv7')(Up1) # (?, 8, 8, 16)
Up2 = UpSampling2D((2,2), interpolation='nearest', name='Up2')(Conv7) # (?, 16, 16, 16)

Conv8 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv8')(Up2) # (?, 16, 16, 32)
Up3 = UpSampling2D((2,2), interpolation='nearest', name='Up3')(Conv8) # (?, 32, 32, 32)

Conv9 = Conv2D(filters[1], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv9')(Up3) # (?, 32, 32, 64)
Up4 = UpSampling2D((2,2), interpolation='nearest', name='Up4')(Conv9) # (?, 64, 64, 64)

Conv10 = Conv2D(filters[0], kernel_size=kernels[1], activation='relu',
               padding='same', name='Conv10')(Up4) # (?, 64, 64, 128)
Up5 = UpSampling2D((2,2), interpolation='nearest', name='Up5')(Conv10) # (?, 128, 128, 128)

Out = Conv2D(b, kernel_size=kernels[0], activation='sigmoid',
             padding='same', name='Output')(Up5) # (?, 128, 128, b)

#%%============================================================================
# Autoencoder Setup

autoencoder = Model(inputs = I, outputs = Out, name='AutoEncoder')
autoencoder.summary()

# Loss definitions for CAE and VCAE models
loss_type = 'binary_crossentropy'
s=1
if VAE:
    losses = KL_Loss(z_mean, z_log_var, side*side*b, loss_type, s) # VCAE
    metrics = [recon_metric(loss_type, side*side*b), KL_metric(z_mean, z_log_var)] # VCAE
else:
    losses =  recon_metric(loss_type, side*side*b)# CAE
    metrics = [recon_metric(loss_type, side*side*b)] # CAE

autoencoder.compile(optimizers.Adam(lr=lr),
                 loss = losses,
                 metrics = metrics)

# Callbacks
checkpoints = ModelCheckpoint(filepath='/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader_weights'+Model_Name+'{epoch:02d}-{loss:.2f}.hdf5',
                              save_weights_only=True, period=25, monitor='loss')

autoencoder.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/untrained_astrovader'+Model_Name+'.h5')

history = autoencoder.fit_generator(train_generator, epochs=Epochs,
                              steps_per_epoch=steps, initial_epoch=0,
                              workers=20, use_multiprocessing=True,
                              callbacks=[checkpoints]
                       )

autoencoder.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader'+Model_Name+'.h5')
autoencoder.save_weights('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader_weights'+Model_Name+'.h5')

np.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/training_history'+Model_Name+'.npy', history)

#%%============================================================================
# Encoder and Decoder Save

encoder = Model(inputs = I, outputs = z, name='Encoder')
encoder.load_weights('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader_weights'+Model_Name+'.h5', by_name=True)
encoder.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_encoder'+Model_Name+'.h5')

decoder_input = Input(shape=(latents,), dtype='float32', name='decoder_input')

index = None
for idx, layer in enumerate(autoencoder.layers):
    if layer.name == 'FC3':
        index = idx
        break

x = autoencoder.get_layer(index=index)(decoder_input)
for i in range(index+1,len(autoencoder.layers)):
    x = autoencoder.get_layer(index=i)(x)
    
decoder = Model(inputs=decoder_input, outputs=x, name='Decoder')
decoder.load_weights('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader_weights'+Model_Name+'.h5', by_name=True)
decoder.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_decoder'+Model_Name+'.h5')