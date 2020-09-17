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
from keras.utils import multi_gpu_model
from keras import Model, optimizers
from keras.callbacks import ModelCheckpoint, Callback

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

def VaDE_Loss(z_mean, z_log_var, gamma, theta_, u_, lambda_, n_centroids, latent_dims, original_dims, loss_type, s=1):
    def vae_loss(y_true, y_pred):

        if loss_type=='binary_crossentropy':
            recon_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*original_dims
        else:
            recon_loss = K.mean(K.square(y_true-y_pred), axis=-1)*original_dims
        
        gmm_loss = 0.5 * K.sum(gamma * K.sum( lambda_ +
                                              K.repeat(K.exp(z_log_var), n_centroids) / K.exp(lambda_) +
                                              K.square( K.repeat(z_mean, n_centroids) - u_) / K.exp(lambda_),
                                              axis=2), axis=1)\
                       - K.sum(gamma * K.log(theta_/gamma), axis=1)\
                       - 0.5 * K.sum(1 + z_log_var, axis=1)

        tot_loss = K.mean(recon_loss + s*gmm_loss, axis=-1)

        return tot_loss
    return vae_loss

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

class AnnealingCallback(Callback):
    def __init__(self, weight, weight_min, weight_max, epoch_start, annealing_time):
        self.weight = weight
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.epoch_start = epoch_start
        self.eps = (weight_max-weight_min)/annealing_time
    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.epoch_start :
            new_weight = min(K.get_value(self.weight) + self.eps, self.weight_max)
            K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))

class MyModelCheckPoint(ModelCheckpoint):

    def __init__(self, singlemodel, *args, **kwargs):
        self.singlemodel = singlemodel
        super(MyModelCheckPoint, self).__init__(*args, **kwargs)
    def on_epoch_end(self, epoch, logs=None):
        self.model = self.singlemodel
        super(MyModelCheckPoint, self).on_epoch_end(epoch, logs)

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

Model_Name = '_Walmsley_'+config['ae_settings']['Model_Name']
VAE = config['ae_settings']['VAE']=='True'
GMM = config['ae_settings']['GMM']=='True'
Epochs = int(config['ae_settings']['Epochs'])
batch_size = int(config['ae_settings']['batch_size'])
n_train = int(config['ae_settings']['n_train'])
steps = n_train//batch_size
side = int(config['ae_settings']['side'])
b = int(config['ae_settings']['b'])
cm = 'grayscale' if b==1 else 'rgb'
In_shape = (side,side,b)
filters = [32,32,16] #from Walmsley et al 2019
kernels = [(3,3), (3,3), (3,3)]
hidden_units = [4096]
latents = int(config['ae_settings']['latents'])
clusters = int(config['ae_settings']['clusters'])
lr = float(config['ae_settings']['lr'])
B_min = 1e-4
B_max = float(config['ae_settings']['beta'])
B = K.variable(B_min, dtype='float32')
b_start = int(config['ae_settings']['beta_start'])
b_time = int(config['ae_settings']['beta_time'])
multi_gpu = config['ae_settings']['multi_gpu']=='True'
pretrain = config['ae_settings']['pretrain'] == 'True'
PT_weights = config['ae_settings']['PT_weights']

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
From Walmsley et al 2019, CAE layout based on BayesCNN, decoder is essentially
the network flipped. FC layer leads to latent layer

More Convs than Cheng et al 2020, but less FC layers and pooling

Question about ordering of upsampling layers. 

ENCODER
In > Conv1 > Conv2 > Pool1 > Conv3 > Conv4 > Pool2 >
     Conv5 > Conv6 > Pool3 > Flat > Latent
     
DECODER
Latent > FC2  > Reshape > Conv7 > Conv8 > Up1 > Conv9 > Conv 10 >
         Up2 > Conv11 > Conv12 > Up3 > Out

"""
#%%============================================================================
# Encoder Setup
I = Input(shape=In_shape, dtype='float32', name='input') # (?, 128, 128, b)

Conv1 = Conv2D(filters[0], kernel_size=kernels[0], activation='relu',
               padding='same', name='Conv1')(I) # (?, 128, 128, 32)
Conv2 = Conv2D(filters[0], kernel_size=kernels[0], activation='relu',
               padding='same', name='Conv2')(Conv1) # (?, 128, 128, 32)
Pool1 = MaxPooling2D((2,2), name='Pool1')(Conv2) # (?, 64, 64, 32)

Conv3 = Conv2D(filters[1], kernel_size=kernels[1], activation='relu',
               padding='same', name='Conv3')(Pool1) # (?, 64, 64, 32)
Conv4 = Conv2D(filters[1], kernel_size=kernels[1], activation='relu',
               padding='same', name='Conv4')(Conv3) # (?, 64, 64, 32)
Pool2 = MaxPooling2D((2,2), name='Pool2')(Conv4) # (?, 32, 32, 32)

Conv5 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv5')(Pool2) # (?, 32, 32, 16)
Conv6 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv6')(Conv5) # (?, 32, 32, 16)
Pool3 = MaxPooling2D((2,2), name='Pool3')(Conv6) # (?, 16, 16, 16)


Flat = Flatten(name='Flat')(Pool3) # (?, 4096)

"""Embedded Layer - Option for Variational or not"""
if VAE:
    z_mean = Dense(latents, name='latentmean')(Flat) # Edit namespace for CAE
    z_log_var = Dense(latents, name='latentlog_var')(Flat) # Comment out for CAE
    z = Lambda(VAE_sampling, output_shape=(latents,),
               name='latentz_sampling')([z_mean,z_log_var]) # Comment out for CAE
else:
    z = Dense(latents, name='latentmean')(Flat) # Edit namespace for CAE

if GMM:    
    z, gamma, t, u, l = GMMLayer(latents, clusters, name='latentGMM_Layer')(z)
#%%============================================================================
# Decoder Setup

FC1 = Dense(hidden_units[0], activation='relu', name='FC1')(z) # (?, 4096)
reshape = Reshape((16,16,16), name='reshape')(FC1) # (?, 16, 16, 16)

Conv7 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv7')(reshape) # (?, 16, 16, 16)
Conv8 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv8')(Conv7) # (?, 16, 16, 16)
Up1 = UpSampling2D((2,2), interpolation='nearest',
                   name='Up1')(Conv8) # (?, 32, 32, 16)

Conv9 = Conv2D(filters[1], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv9')(Up1) # (?, 32, 32, 32)
Conv10 = Conv2D(filters[1], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv10')(Conv9) # (?, 32, 32, 32)
Up2 = UpSampling2D((2,2), interpolation='nearest',
                   name='Up2')(Conv10) # (?, 64, 64, 32)

Conv11 = Conv2D(filters[0], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv11')(Up2) # (?, 64, 64, 32)
Conv12 = Conv2D(filters[0], kernel_size=kernels[2], activation='relu',
               padding='same', name='Conv12')(Conv11) # (?, 64, 64, 32)
Up3 = UpSampling2D((2,2), interpolation='nearest',
                   name='Up3')(Conv12) # (?, 128, 128, 32)

Out = Conv2D(b, kernel_size=kernels[0], activation='sigmoid',
             padding='same', name='Output')(Up3) # (?, 128, 128, b)

#%%============================================================================
# Autoencoder Setup

autoencoder = Model(inputs = I, outputs = Out, name='AutoEncoder')
autoencoder.summary()

# Loss definitions for CAE and VCAE models
loss_type = 'binary_crossentropy'
if GMM:
    losses = VaDE_Loss(z_mean, z_log_var, gamma, t, u, l, clusters, latents,
                       side*side*b, loss_type, B)
    metrics = [recon_metric(loss_type, side*side*b),
                VaDE_metric(z_mean, z_log_var, gamma, t, u, l, clusters,
                            latents)]
elif VAE:
    losses = KL_Loss(z_mean, z_log_var, side*side*b, loss_type, B) # VCAE
    metrics = [recon_metric(loss_type, side*side*b),
               KL_metric(z_mean, z_log_var)] # VCAE
else:
    losses =  recon_metric(loss_type, side*side*b)# CAE
    metrics = [recon_metric(loss_type, side*side*b)] # CAE

autoencoder.compile(optimizers.Adam(lr=lr),
                 loss = losses,
                 metrics = metrics)

# Callbacks
checkpoints = MyModelCheckPoint(autoencoder, filepath='/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader_weights'+Model_Name+'{epoch:02d}-{loss:.2f}.h5',
                              save_weights_only=True, period=25, monitor='loss')
annealing = AnnealingCallback(B, B_min, B_max, epoch_start=b_start, annealing_time=b_time)


autoencoder.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/untrained_astrovader'+Model_Name+'.h5')

if pretrain:
    autoencoder.load_weights(PT_weights, by_name=True)

if multi_gpu:
    ae_par = multi_gpu_model(autoencoder, gpus=2)

    ae_par.compile(optimizers.Adam(lr=lr),
                   loss = losses,
                   metrics = metrics)

    history = ae_par.fit_generator(train_generator, epochs=Epochs,
                              steps_per_epoch=steps, initial_epoch=0,
                              workers=20, use_multiprocessing=True,
                              callbacks=[checkpoints, annealing]
                       )
else:
    history = autoencoder.fit_generator(train_generator, epochs=Epochs,
                              steps_per_epoch=steps, initial_epoch=0,
                              workers=20, use_multiprocessing=True,
                              callbacks=[checkpoints, annealing]
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
    if layer.name == 'FC1':
        index = idx
        break

x = autoencoder.get_layer(index=index)(decoder_input)
for i in range(index+1,len(autoencoder.layers)):
    x = autoencoder.get_layer(index=i)(x)
    
decoder = Model(inputs=decoder_input, outputs=x, name='Decoder')
decoder.load_weights('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_astrovader_weights'+Model_Name+'.h5', by_name=True)
decoder.save('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_decoder'+Model_Name+'.h5')