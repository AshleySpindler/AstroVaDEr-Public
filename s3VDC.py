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
    First, the VAE is trained using standard
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
import keras.backend as K
from keras import Model, optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.utils import multi_gpu_model
import tensorflow as tf

from AV_Losses import gamma_training, beta_training, static_training, KL_metric, recon_metric, VaDE_metric, log_prob
from AV_Callbacks import lr_schedule, AnnealingCallback, MyModelCheckPoint
from AV_Models import encoder_setup, embedding_setup, decoder_setup

import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.mixture as mixture

if __name__ == "__main__":
    #%%============================================================================
    # Initial Setup
    
    """ Settings
    Settings are read from a text file via configparser
    Model name is set by user, + a time index so we don't accidentily overwrite
    old model weights
    Directories should point to train/test/valid directories that contain the
    processed images inside another directory for keras data generator, i.e.:
        /path/to/train/train/(images)
        train_dir = '/path/to/train/'
    filters and kernels currently set in script because reading lists is a pain
    
    params dictionary is passed to various functions, so it contains all the
    model settings and such:
        batch_size, n_train/test, epochs, ect: self explanetory
        gmm_steps, warm_up_steps, _annealing_steps,
        static_steps, annealing_periods: S3VDC parameters, see paper of Cao et
        al 2020 reference within for details
        l2_regularizer: regularising factor put on the weights and biases of
        the convolutional blocks, etc.
        GPUs: how many GPUs to use in training
        side, bands, etc refer to image dimensions, i.e.: 128x128x1
        clusters, latents are the number of GMM components and latent variables
        lr_settings set up the lr and decay parameters
        warm up is the KL weight during the warm up training phase
        annealing factor is a keras variable that is updated during the training
        of the gmm parameters
        loss type and output activity depend on the type of data, MSE seems to
        work best for galaxy images
        scale jank is an additional weighting on the GMM loss during training,
        because there's still some weirdness in the relative weighting of the
        clustering a reconstructions losses
    """
    
    config = configparser.ConfigParser()
    config.read('s3VDC-config.txt')
    
    Model_Name = config['directories']['Model_Name']+'-'+str(time.time())
    train_dir = config['directories']['train_dir']
    valid_dir = config['directories']['valid_dir']
    model_dir = config['directories']['model_dir']
    filters = [64,64,16]
    kernels = [(3,3), (5,5), (5,5)]
    flat_units = [4096]
    
    params = {}
    
    params['batch_size'] = int(config['training']['batch_size'])
    n_train = int(config['training']['n_train'])
    steps_per_epoch = n_train//params['batch_size']
    params['gmm_steps'] = int(config['training']['gmm_steps'])
    params['warm_up_steps'] = int(config['training']['warm_up_steps'])
    params['annealing_steps'] = int(config['training']['annealing_steps'])
    params['static_steps'] = int(config['training']['static_steps'])
    params['annealing_periods'] = int(config['training']['annealing_periods'])
    params['total_steps'] = params['warm_up_steps'] + params['annealing_periods']\
                           * (params['annealing_steps'] + params['static_steps'])
    params['l2_regularizer'] = float(config['training']['l2_regulariser'])
    params['GPUs'] = int(config['training']['GPUs'])
    
    side = int(config['dataset']['side'])
    b = params['bands'] = int(config['dataset']['b'])
    cm = 'grayscale' if b==1 else 'rgb'
    In_shape = (side,side,b)
    params['original_dims'] = side*side*b
    
    params['latents'] = int(config['embedding']['latents'])
    params['clusters'] = int(config['embedding']['clusters'])
    
    params['lr'] = float(config['lr_settings']['lr'])
    params['min_lr'] = float(config['lr_settings']['min_lr'])
    params['lr_steps'] = int(config['lr_settings']['lr_steps'])
    params['lr_decay'] = np.power(params['min_lr']/params['lr'],
                                  params['lr_steps']/params['total_steps'])
    
    params['warm_up'] = float(config['loss_settings']['warm_up_factor'])
    params['annealing_factor'] = K.variable(float(config['loss_settings']['warm_up_factor']), dtype='float32')
    params['loss_type'] = config['loss_settings']['loss_type']
    params['output_activity'] = 'sigmoid' if params['loss_type']=='binary_crossentropy' else 'relu'
    params['scale_jank'] = config['loss_settings']['scale_jank']
    
    #%%============================================================================
    # Create image data generator - rotates images and improves contrast
    
    print('Loading data sets')
    
    datagen = ImageDataGenerator(rescale=1./255,
                                 horizontal_flip=True, vertical_flip=True,
                                 dtype='float32', fill_mode='wrap')
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
    print('Setting up models')
    enc_in, enc_out = encoder_setup(In_shape, filters, kernels, params)
    outputs, gmm_weights = embedding_setup(params, enc_out)
    dec_in, dec_out = decoder_setup(flat_units, filters, kernels,
                                    params, dec_in=outputs['z_out'])
    with tf.device('/cpu:0'):
        # we set these models to be on the CPU initially so theres no conflict
        # on the GPU memory
        enc_vade = Model(inputs=enc_in, outputs=outputs['z_mean'], name='VADE_Encoder')
        
        VADE = Model(inputs=enc_in, outputs=dec_out, name='VADE')
        
        # Freeze GMM Layer during y-training phase
        VADE.get_layer('latentGMM_Layer').trainable = False
                
        # Callbacks
        
        AnnealingCB = AnnealingCallback(params)
        lr_decayCB = LearningRateScheduler(lr_schedule(params['lr'], params['min_lr'], params['lr_steps'], params['lr_decay']), verbose=1)
        tb_logs = TensorBoard(log_dir=model_dir+'/tb_logs/'+Model_Name,
                              histogram_freq=1, batch_size=params['batch_size'],
                              write_graph=False)
        checkpoints = MyModelCheckPoint(VADE, filepath=model_dir+Model_Name+'{epoch:02d}-{loss:.2f}.h5',
                                  save_weights_only=True, period=10, monitor='loss')
    
    # Losses and Metrics
    
    warm_up_loss = gamma_training(outputs, params)
    beta_loss = beta_training(outputs, gmm_weights, params)
    static_loss = static_training(outputs, gmm_weights, params)
    vae_loss = KL_metric(outputs)
    recon_loss = recon_metric(params)
    vade_loss = VaDE_metric(outputs, gmm_weights, params)
    log_prob_met = log_prob(outputs['z_out'], gmm_weights, params)

    # Parallelize and Compile
    if params['GPUs'] > 1:
        VADE_gpu = multi_gpu_model(VADE, gpus=params['GPUs'])
    else:
        VADE_gpu = VADE
        
    VADE_gpu.compile(optimizers.Adam(lr=params['lr']),
                loss = warm_up_loss,
                metrics = [recon_loss,vae_loss])
    
    #%%============================================================================
    # y-Training
    
    # Initially train VAE for Ty epochs using vanilla KL loss
    print('========================== y-Training Phase ==========================')
    print('======================================================================')
    print('\n')
    print('Training for {0} epochs with KL weight {1}'.
          format(params['warm_up_steps'], params['warm_up']))
          
    warm_up_hist = VADE_gpu.fit_generator(train_generator, epochs=params['warm_up_steps'],
                                  steps_per_epoch=steps_per_epoch, initial_epoch=0,
                                  callbacks=[lr_decayCB, tb_logs, checkpoints],
                                  validation_data=(x_test, x_test),
                                  use_multiprocessing=True, max_queue_size=1000,
                           )
    
    VADE.save_weights(model_dir+Model_Name+'warm_up_weights.h5')
    np.save(model_dir+Model_Name+'warm_up_losses.npy', warm_up_hist.history)
    
    x_pred = VADE_gpu.predict(x_test[0:100])
    f, ax = plt.subplots(10,10)
    [[ax.ravel()[j].imshow(x_pred[j,:,:,0], vmin=0, vmax=1, cmap='binary') for j in range(100)]]
    plt.show()
    plt.close()
    
    #%%============================================================================
    # mini-batch GMM initialization
    
    # Collect k batches of predicted samples to fit GMM
    
    print('==================== Mini-batch GMM initialization ====================')
    print('=======================================================================')
    print('\n')
    print('Collecting {0} mini-batches for GMM fit'.format(params['gmm_steps']))
    
    data, _ = train_generator.next()
    gmm_data = enc_vade.predict(data)
    for i in range(0, params['gmm_steps']-1):
        data, _ = train_generator.next()
        gmm_data = np.append(gmm_data, enc_vade.predict(data), axis=0)
    
    print('Mini-batches collected')
    print('\n')
    print('Performing K-means clustering to find cluster centers')
    
    kmeans = cluster.KMeans(n_clusters=params['clusters'], random_state=0)
    kmeans.fit(gmm_data)
    
    print('\n')
    print('Performing Gaussian Mixture Modelling to find cluster covariances')
    
    gmm = mixture.GaussianMixture(n_components=params['clusters'],
                                  covariance_type='diag',
                                  max_iter=10000,
                                  means_init=kmeans.cluster_centers_,
                                  random_state=100,
                                  verbose=1)
    gmm.fit(gmm_data)
    
    mtm = (1-params['warm_up'])**3 # momentum factor to update GMM layer weights
    
    print('\n')
    print('Updating GMM weights in Model with momentum {0:.3e}'.format(mtm))
    
    K.set_value(gmm_weights['theta'], (1.0 - mtm) * K.get_value(gmm_weights["theta"]) + mtm * gmm.weights_.T)
    K.set_value(gmm_weights['mu'], (1.0 - mtm) * K.get_value(gmm_weights["mu"]) + mtm * gmm.means_.T)
    K.set_value(gmm_weights['lambda'], (1.0 - mtm) * K.get_value(gmm_weights["lambda"]) + mtm * gmm.covariances_.T)
    
    #%%============================================================================
    # Beta Annealing and Static Training
    
    print('================= Beta Annealing and Static Training =================')
    print('======================================================================')
    print('\n')
    
    #Unfreeze GMM layer and set variables for lr and cond_z layer
        
    lr_new = params['lr'] * np.power(params['lr_decay'],
                            np.floor(params['warm_up_steps']/params['lr_steps']))
    
    print('Unfreezing GMM Layer for Training')
    VADE.get_layer('latentGMM_Layer').trainable = True # affects VAE_gpu
    
    # recompile model to set trainable weights and new loss function
    VADE_gpu.compile(optimizers.Adam(lr=params['lr'], clipvalue=1),
                loss = beta_loss,
                metrics = [beta_loss,static_loss,recon_loss,
                           vae_loss,vade_loss,log_prob_met])
    
    total_epochs = params['total_steps']
    
    print('\n')
    print('Training for {0} epochs, consisting of {1} Annealing/Static Phases'.
          format(total_epochs, params['annealing_periods']))
    print('Annealing phases last {0} epochs and Static Phases last {1} epochs'.
          format(params['annealing_steps'], params['static_steps']))
    
    annealing_hist = VADE_gpu.fit_generator(train_generator, epochs=total_epochs,
                                  steps_per_epoch=steps_per_epoch, initial_epoch=params['warm_up_steps']+1,
                                  callbacks=[AnnealingCB, lr_decayCB, checkpoints, tb_logs],
                                  validation_data=(x_test, x_test),
                                  use_multiprocessing=True, max_queue_size=1000,
                           )
    
    VADE.save_weights(model_dir+Model_Name+'final_weights.h5')
    np.save(model_dir+Model_Name+'annealing_losses.npy', annealing_hist.history)
    
    x_pred = VADE_gpu.predict(x_test[0:100])
    f, ax = plt.subplots(10,10)
    [[ax.ravel()[j].imshow(x_pred[j,:,:,0], vmin=0, vmax=1, cmap='binary') for j in range(100)]]
    plt.show()
    plt.close()
    
    
    print('Training Complete')