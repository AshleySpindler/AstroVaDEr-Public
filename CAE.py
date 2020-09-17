#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:04:20 2020

@author: ashleyspindler

AstroVaDEr implementation of "simple, scalable, and stable VDC" (s3VDC),
as presented in Cao et al 2020 (arXiv preprint arXiv:2005.08047).

CAE networks based on CNN developed in Walmsley et al 2019 (GalZoo2)

VaDE framework developed based on Jiang et al 2016 arXiv:1611.05148

Implementation uses keras data generators to feed images (rgb or grayscale)
from hard disk. This script uses 128x128xB images from Galaxy Zoo, which are
stored as 256x256x3 uint PNGs.

Training consists of four phases:
    First, the CAE is trained using standard
    KL divergence loss based on single Gaussian (gamma_training) with a small
    KL weights (y) compared to reconstruction loss (mse or bxe supported).
    
    Second, the GMM aparmeters of VaDE are initialised by using sklearn KMeans
    and GaussianMixture methods. Cao 2020 shows that good GMM fit can be achieved
    using small fraction of mini-batches from the full training set (kxL << N).
    
    Third phase introduces the full VaDE loss function between the latent space
    and a GMM prior. Martin et al 2020(?) shows that unsupervised clustering
    works well if using large number of clusters, as does Cheng et al 2020. We
    use 128 clusters for 128 latents (based on Walmsley CNN) for initial tests.
    A polynomial function is used to anneal the KL weight for a fixednumber of
    epochs, until it reaches y+1.
    
    Finally, the KL weight is set to 1 for a fixed number of epochs for the
    static training phase. The beta annealing and static training phases are
    repeated M times, which improves disentanglement in the latent space while
    maintaining reconstruction quality.
    


"""

import configparser #used to read config file
import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape,\
                         MaxPooling2D, UpSampling2D, Conv2D,\
                         GaussianNoise, LeakyReLU
from keras import Model, optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.utils import multi_gpu_model
import tensorflow as tf

from AV_Losses import recon_metric
from AV_Callbacks import lr_schedule, MyModelCheckPoint

import matplotlib.pyplot as plt

def encoder_setup(In_Shape, filters, kernels, params):
    I = Input(shape=In_shape, dtype='float32', name='encoder_input') # (?, 128, 128, b)
    Noise = GaussianNoise(1e-8)(I)
    
    Conv1 = Conv2D(filters[0], kernel_size=kernels[0],# activation='relu',
                   padding='same', name='Conv1',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Noise) # (?, 128, 128, 32)
    Conv1 = LeakyReLU(alpha=0.1)(Conv1)
    Conv2 = Conv2D(filters[0], kernel_size=kernels[0],# activation='relu',
                   padding='same', name='Conv2',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv1) # (?, 128, 128, 32)
    Conv2 = LeakyReLU(alpha=0.1)(Conv2)
    Pool1 = MaxPooling2D((2,2), name='Pool1')(Conv2) # (?, 64, 64, 32)
    
    Conv3 = Conv2D(filters[1], kernel_size=kernels[1],# activation='relu',
                   padding='same', name='Conv3',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Pool1) # (?, 64, 64, 32)
    Conv3 = LeakyReLU(alpha=0.1)(Conv3)
    Conv4 = Conv2D(filters[1], kernel_size=kernels[1],# activation='relu',
                   padding='same', name='Conv4',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv3) # (?, 64, 64, 32)
    Conv4 = LeakyReLU(alpha=0.1)(Conv4)
    Pool2 = MaxPooling2D((2,2), name='Pool2')(Conv4) # (?, 32, 32, 32)
    
    Conv5 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv5',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Pool2) # (?, 32, 32, 16)
    Conv5 = LeakyReLU(alpha=0.1)(Conv5)
    Conv6 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv6',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv5) # (?, 32, 32, 16)
    Conv6 = LeakyReLU(alpha=0.1)(Conv6)
    Pool3 = MaxPooling2D((2,2), name='Pool3')(Conv6) # (?, 16, 16, 16)
    
    
    Flat = Flatten(name='Flat')(Pool3) # (?, 4096)

    return I, Flat

def embedding_setup(params, enc_out):
    z_mean = Dense(params['latents'], name='latentmean',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(enc_out) # Edit namespace for CAE
        
    return z_mean

def decoder_setup(flat_units, filters, kernels, params, dec_in=None):
    
    if dec_in is None:
        dec_in = Input(shape=(params['latents'],), dtype='float32', name='decoder_input')

    FC3 = Dense(flat_units[0], name='FC3',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(dec_in) # (?, 4096)
    FC3 = LeakyReLU(alpha=0.1)(FC3)
    reshape = Reshape((16,16,filters[2]), name='reshape')(FC3) # (?, 16, 16, 16)
    
    Conv7 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv7',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(reshape) # (?, 16, 16, 16) 
    Conv7 = LeakyReLU(alpha=0.1)(Conv7)
    Conv8 = Conv2D(filters[2], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv8',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv7) # (?, 16, 16, 16) 
    Conv8 = LeakyReLU(alpha=0.1)(Conv8)
    Up1 = UpSampling2D((2,2), interpolation='bilinear',
                       name='Up1')(Conv8) # (?, 32, 32, 16)
    
    Conv9 = Conv2D(filters[1], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv9',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Up1) # (?, 32, 32, 32) 
    Conv9 = LeakyReLU(alpha=0.1)(Conv9)
    Conv10 = Conv2D(filters[1], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv10',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv9) # (?, 32, 32, 32) 
    Conv10 = LeakyReLU(alpha=0.1)(Conv10)
    Up2 = UpSampling2D((2,2), interpolation='bilinear',
                       name='Up2')(Conv10) # (?, 64, 64, 32)
    
    Conv11 = Conv2D(filters[0], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv11',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Up2) # (?, 64, 64, 32) 
    Conv11 = LeakyReLU(alpha=0.1)(Conv11)
    Conv12 = Conv2D(filters[0], kernel_size=kernels[2],# activation='relu',
                   padding='same', name='Conv12',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv11) # (?, 64, 64, 32) 
    Conv12 = LeakyReLU(alpha=0.1)(Conv12)
    Up3 = UpSampling2D((2,2), interpolation='bilinear',
                       name='Up3')(Conv12) # (?, 128, 128, 32)
    
    Out = Conv2D(b, kernel_size=kernels[0], activation=params['output_activity'],
                 padding='same', name='Output',
                 kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                 bias_regularizer=regularizers.l2(params['l2_regularizer']),
                 )(Up3) # (?, 128, 128, b)
    
    return dec_in, Out


if __name__ == "__main__":
    #%%============================================================================
    # Initial Setup
    
    """ Hyperparameters
    TODO
    """
    
    config = configparser.ConfigParser()
    config.read('CAE-config.txt') 
    
    Model_Name = 'CAE_'+str(time.time())+'_'
    train_dir = config['directories']['train_dir']
    valid_dir = config['directories']['valid_dir']
    model_dir = config['directories']['model_dir']+'CAE/'
    filters = [64,64,16]
    kernels = [(3,3), (5,5), (5,5)]
    flat_units = [4096]
    
    params = {}
    
    params['batch_size'] = int(config['training']['batch_size'])
    n_train = int(config['training']['n_train'])
    steps_per_epoch = n_train//params['batch_size']
    params['total_steps'] = int(config['training']['total_steps'])
    params['l2_regularizer'] = float(config['training']['l2_regulariser'])
    
    side = int(config['dataset']['side'])
    b = int(config['dataset']['b'])
    cm = 'grayscale' if b==1 else 'rgb'
    In_shape = (side,side,b)
    params['original_dims'] = side*side*b
    
    params['latents'] = int(config['embedding']['latents'])
    
    params['lr'] = float(config['lr_settings']['lr'])
    params['min_lr'] = float(config['lr_settings']['min_lr'])
    params['lr_steps'] = int(config['lr_settings']['lr_steps'])
    params['lr_decay'] = np.power(params['min_lr']/params['lr'],
                                  params['lr_steps']/params['total_steps'])
    
    params['loss_type'] = config['loss_settings']['loss_type']
    params['output_activity'] = 'sigmoid' if params['loss_type']=='binary_crossentropy' else 'relu'
    
    #%%============================================================================
    # Create image data generator - rotates images and improves contrast
    
    datagen = ImageDataGenerator(rescale=1./255,#zoom_range=(0.75,0.75),
                                 horizontal_flip=True, vertical_flip=True,
                                 dtype='float32', fill_mode='wrap',
                                 preprocessing_function=None)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(side,side),
                                                  color_mode=cm, class_mode='input',
                                                  batch_size=params['batch_size'],
                                                  shuffle=True)
    valid_generator = datagen.flow_from_directory(valid_dir, target_size=(side,side),
                                                  color_mode=cm, class_mode='input',
                                                  batch_size=200,#params['batch_size'],
                                                  shuffle=True)
    x_test = np.zeros((5000,128,128,1), dtype='float32')
    for i in range(25):
        x_test[i*200:(i+1)*200], _ = valid_generator.next()    
    
    #%%============================================================================
    # Model Setup
    
    with tf.device('/cpu:0'):
        enc_in, enc_out = encoder_setup(In_shape, filters, kernels, params)
        z = embedding_setup(params, enc_out)
        dec_in, dec_out = decoder_setup(flat_units, filters, kernels,
                                        params, dec_in=z)
        
        enc_cae = Model(inputs=enc_in, outputs=z, name='CAE_Encoder')
        
        CAE = Model(inputs=enc_in, outputs=dec_out, name='CAE')
        
        # Losses and Metrics
        recon_loss = recon_metric(params)
        
        # Callbacks
        
        lr_decayCB = LearningRateScheduler(lr_schedule(params['lr'], params['min_lr'], params['lr_steps'], params['lr_decay']), verbose=1)
        tb_logs = TensorBoard(log_dir=model_dir+'/tb_logs/'+Model_Name,
                              histogram_freq=1, batch_size=params['batch_size'],
                              write_graph=False)
        checkpoints = MyModelCheckPoint(CAE, filepath=model_dir+Model_Name+'{epoch:02d}-{loss:.2f}.h5',
                                  save_weights_only=True, period=10, monitor='loss')
    # Parallelize and Compile
    
    CAE_gpu = multi_gpu_model(CAE, gpus=2)
    
    CAE_gpu.compile(optimizers.Adam(lr=params['lr']),
                loss = recon_loss)
    
    #%%============================================================================
    # y-Training
    
    # Initially train VAE for Ty epochs using vanilla KL loss
    print('========================== Training Phase ==========================')
    print('====================================================================')
    
    losses_hist = CAE_gpu.fit_generator(train_generator, epochs=params['total_steps'],
                                  steps_per_epoch=steps_per_epoch, initial_epoch=0,
                                  callbacks=[lr_decayCB, tb_logs, checkpoints],
                                  validation_data=(x_test, x_test), max_queue_size=1000,
                                  use_multiprocessing=True,
                           )
    
    x_pred = CAE_gpu.predict(x_test[0:100])
    f, ax = plt.subplots(10,10)
    [[ax.ravel()[j].imshow(x_pred[j,:,:,0], vmin=0, vmax=1, cmap='binary') for j in range(100)]]
    plt.show()
    plt.close()

    CAE.save_weights(model_dir+Model_Name+'_weights-.h5')
    np.save(model_dir+Model_Name+'_losses.npy', losses_hist.history)
        
    print('Training Complete')
