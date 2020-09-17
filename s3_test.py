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

from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras import Model
from keras.utils import multi_gpu_model
import tensorflow as tf

from AV_Models import encoder_setup, embedding_setup, decoder_setup

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

if __name__ == "__main__":
    #%%============================================================================
    # Initial Setup
    
    config = configparser.ConfigParser()
    config.read('s3VDC-config.txt') 
    
    saved_weights = config['directories']['saved_weights']
    test_dir = config['directories']['test_dir']
    model_dir = config['directories']['model_dir']
    filters = [64,64,16]
    kernels = [(3,3), (5,5), (5,5)]
    flat_units = [4096]
    
    params = {}
    
    params['batch_size'] = int(config['training']['batch_size'])
    n_test = int(config['training']['n_test'])
    steps_per_epoch = n_test//params['batch_size']
    params['l2_regularizer'] = float(config['training']['l2_regulariser'])
    params['GPUs'] = int(config['training']['GPUs'])
    
    side = int(config['dataset']['side'])
    b = int(config['dataset']['b'])
    cm = 'grayscale' if b==1 else 'rgb'
    In_shape = (side,side,b)
    params['original_dims'] = side*side*b
    
    params['latents'] = int(config['embedding']['latents'])
    params['clusters'] = int(config['embedding']['clusters'])
        
    params['warm_up'] = float(config['loss_settings']['warm_up_factor'])
    params['annealing_factor'] = K.variable(float(config['loss_settings']['warm_up_factor']), dtype='float32')
    params['loss_type'] = config['loss_settings']['loss_type']
    params['output_activity'] = 'sigmoid' if params['loss_type']=='binary_crossentropy' else 'relu'
    
    #%%============================================================================
    # Create image data generator - rotates images and improves contrast
    print('datagen')
    datagen = ImageDataGenerator(rescale=1./255, dtype='float32')
    test_generator = datagen.flow_from_directory(test_dir, target_size=(side,side),
                                                  color_mode=cm, class_mode=None,
                                                  batch_size=params['batch_size'],
                                                  shuffle=True)
    x_test = np.zeros((n_test,128,128,1), dtype='float32')
    for i in range(steps_per_epoch):
        x_test[i*params['batch_size']:(i+1)*params['batch_size']] = test_generator.next()
        
    #%%============================================================================
    # Model Setup
    
    enc_in, enc_out = encoder_setup(In_shape, filters, kernels, params)
    outputs, gmm_weights = embedding_setup(params, enc_out)
    dec_in, dec_out = decoder_setup(flat_units, filters, kernels,
                                    params, dec_in=outputs['z_out'])
    dec_in2, dec_out2 = decoder_setup(flat_units, filters, kernels,
                                    params)
    with tf.device('/cpu:0'):        
        VADE = Model(inputs=enc_in, outputs=dec_out, name='VADE')
    
    encoder = Model(inputs=enc_in, outputs=(outputs['z_mean'],
                                        outputs['z_log_var'],
                                        outputs['z'],
                                        outputs['gamma']),
                name='VADE_Encoder')
    
    decoder = Model(inputs=dec_in2, outputs=dec_out2)
    
    # Parallelize
    
    if params['GPUs'] > 1:
        VADE_gpu = multi_gpu_model(VADE, gpus=params['GPUs'])
    else:
        VADE_gpu = VADE
    
    # Load trained weights
    VADE.load_weights(model_dir+saved_weights)
    encoder.load_weights(model_dir+saved_weights, by_name=True)
    decoder.load_weights(model_dir+saved_weights, by_name=True)
    
    # Predict embeddings and image reconstructions
    z_mu, z_ls, z, y = encoder.predict(x_test, batch_size=100, verbose=True)
    c_v = np.argmax(y, axis=1)
    labels, counts = np.unique(c_v, return_counts=True)
    x_pred = VADE_gpu.predict(x_test, batch_size=200, verbose=True)
    
    # Access GMM weights
    mu = K.get_value(gmm_weights['mu'])
    sigma = K.get_value(gmm_weights['lambda'])
    theta = K.get_value(gmm_weights['theta'])
    
    #%%============================================================================
    # Plot some random reconstructions with residuals
    gals = np.random.choice(n_test, 60, replace=False)
    fig = plt.figure(figsize=(18, 18))
    
    outer_grid = gridspec.GridSpec(5, 1, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    
    k = 0
    for i in range(0,5):
        inner_grid = gridspec.GridSpecFromSubplotSpec(3, 12,
            subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        for j in range(0,12):
            g = gals[k]
            
            ax = plt.Subplot(fig, inner_grid[0,j])
            ax.imshow(x_test[g,:,:,0], vmin=0, vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            
            ax = plt.Subplot(fig, inner_grid[1,j])
            ax.imshow(x_pred[g,:,:,0], vmin=0, vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            
            ax = plt.Subplot(fig, inner_grid[2,j])
            ax.imshow((x_test[g,:,:,0]-x_pred[g,:,:,0]), cmap='binary')
            k+=1
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    
    plt.show()
    
    #%%============================================================================
    # Plot reconstructions of best and worst reconstructed images
    mse = np.mean(np.square(x_test-x_pred), axis=(1,2,3))
    mse_ord = np.argsort(mse)
    indx = [mse_ord[5:10], mse_ord[-6:-1]]
    
    fig = plt.figure(figsize=(18, 18))
    
    outer_grid = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 5,
            subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        imgs_t = x_test[indx[i]]
        imgs_p = x_pred[indx[i]]
        for j in range(0,5):
            ax = plt.Subplot(fig, inner_grid[0,j])
            ax.imshow(imgs_t[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            ax = plt.Subplot(fig, inner_grid[1,j])
            ax.imshow(imgs_p[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Cluster analysis
    import sklearn.cluster as cluster
    import sklearn.mixture as mixture
    import sklearn.metrics as metrics
    
    kmeans = cluster.KMeans(n_clusters=params['clusters'], random_state=0)
    kmeans.fit(z_mu)
    c_k = kmeans.predict(z_mu)
    
    gmm = mixture.GaussianMixture(n_components=params['clusters'],
                                  covariance_type='diag',
                                  max_iter=10000,
                                  means_init=kmeans.cluster_centers_,
                                  random_state=100,
                                  verbose=1)
    gmm.fit(z_mu)
    c_g = gmm.predict(z_mu)
    
    CH_scores = (metrics.calinski_harabasz_score(z_mu, c_v),
                 metrics.calinski_harabasz_score(z_mu, c_k),
                 metrics.calinski_harabasz_score(z_mu, c_g))
    
    SS_scores = (metrics.silhouette_score(z_mu, c_v),
                 metrics.silhouette_score(z_mu, c_k),
                 metrics.silhouette_score(z_mu, c_g))
    
    print(CH_scores)
    print(SS_scores)
    
    mixture_overlap = np.zeros((12,12))
    for i in range(12):
        mixture_overlap[i,:] = np.mean(y[c_v==i], axis=0)
        plt.plot(mixture_overlap[i,:], lw=2, label='Primary Cluster {0}'.format(i))
        
    plt.xlabel('Secondary Clusters')
    plt.ylabel('Mean Probability')
    plt.title('Cluster Overlap Probabilities')
    plt.tight_layout()
    plt.show()
    
    #%%============================================================================
    # Plot highest prob gals for each component - predictions
    
    fig = plt.figure(figsize=(18, 18))
        
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        imgs = x_pred[np.argsort(-y[:,labels[i]])[0:45]]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Plot highest prob gals in each conponent - truth
    
    fig = plt.figure(figsize=(18, 18))
        
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        imgs = x_test[np.argsort(-y[:,labels[i]])[0:45]]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Plot random gals assigned in each cluster - predictions
    
    gals = []
    
    fig = plt.figure(figsize=(18, 18))
        
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        gals.append(np.random.choice(counts[i], 45, replace=False))
        imgs = x_pred[c_v==labels[i]][gals[i]]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Plot random gals assigned in each cluster - truth
    
    fig = plt.figure(figsize=(18, 18))
    
    labels = np.unique(c_v)
    
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        imgs = x_test[c_v==labels[i]][gals[i]]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Plot highest prob gals in each cluster with labels - predictions
    
    fig = plt.figure(figsize=(18, 18))
        
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        Y = y[c_v==i][:,labels[i]]
        X = x_pred[c_v==i]
        imgs = X[np.argsort(-Y)][0:45]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Plot highest prob gals in each cluster with labels - truth
    
    fig = plt.figure(figsize=(18, 18))
        
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        Y = y[c_v==i][:,labels[i]]
        X = x_test[c_v==i]
        imgs = X[np.argsort(-Y)][0:45]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    fig = plt.figure(figsize=(18, 18))
    
    labels_r = [0, 1, 2, 3, 4]
    c_r = np.argmax(y[:,(2,3,7,8,10)], axis=1)
    gals_r = np.where((c_v==0)|(c_v==1)|(c_v==4)|(c_v==5)|(c_v==6)|(c_v==11))[0]
    c_r = c_r[gals_r]
    y_r = y[:,(2,3,7,8,10)][gals_r]
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(3, 2, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(5):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        Y = y_r[c_r==i][:,labels_r[i]]
        X = x_pred[gals_r][c_r==i]
        imgs = X[np.argsort(-Y)][0:45]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    fig = plt.figure(figsize=(18, 18))
    
    labels_r = [0, 1, 2, 3, 4]
    c_r = np.argmax(y[:,(2,3,7,8,10)], axis=1)
    gals_r = np.where((c_v==0)|(c_v==1)|(c_v==4)|(c_v==5)|(c_v==6)|(c_v==11))[0]
    c_r = c_r[gals_r]
    y_r = y_r[gals_r]
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(3, 2, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(5):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        Y = y_r[c_r==i][:,labels_r[i]]
        X = x_test[gals_r][c_r==i]
        imgs = X[np.argsort(-Y)][0:45]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    # Image Generation by cluster component
    
    def z_generation(_mu, _sigma, N):
        
        return np.random.multivariate_normal(_mu, _sigma*np.identity(len(_sigma)), N)
    
    
    z_gen = np.zeros((45*params['clusters'], params['latents']), dtype='float32')
    
    for i in range(0,12):
        z_gen[i*45:(i+1)*45] = z_generation(mu[:,i], sigma[:,i]/2, 45)
        
    x_gen = decoder.predict(z_gen, batch_size=90, verbose=1)
    
    fig = plt.figure(figsize=(18, 18))
    
    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.1, top=0.99, right=0.99, bottom=0.01, left=0.01)
    for i in range(12):
        inner_grid = gridspec.GridSpecFromSubplotSpec(5, 9,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        
        imgs = x_gen[i*45:(i+1)*45]
        for j, (c, d) in enumerate(product(range(1, 10), repeat=2)):
            if j>=45:
                continue
            ax = plt.Subplot(fig, inner_grid[j])
            try:
                ax.imshow(imgs[j,:,:,0], vmin=0, vmax=1, cmap='gray')
            except IndexError:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            
    all_axes = fig.get_axes()
    
    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()
    
    #%%============================================================================
    #Comparisons between VaDE clustering and GalZoo 2
    
    # load in IDs of test galaxies so we can cross compare them with GZ2
    from astropy.io import fits
    try:
        gz_test = fits.open('data/gz2_hart16.fits.gz')[1].data
    except FileNotFoundError:
        import wget
        wget.download(url='http://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/gz2_hart16.fits.gz',
                      out='data/gz2_hart16.fits.gz')
        gz_test = fits.open('data/gz2_hart16.fits.gz')[1].data

    test_ids = np.load('data/TestGals_IDs.npy')
    # discard gals not in test set
    gz_test = gz_test[np.where(np.isin(gz_test['dr7objid'],test_ids))[0]]
    # sort both files into same order
    test_ids = test_ids[np.argsort(test_ids)]
    gz_test = gz_test[np.argsort(gz_test['dr7objid'])]
    
    # need to reload the test images so they are in the same order
    x_test = np.zeros((len(test_ids),128,128,1))
    import cv2
    for i in range(len(test_ids)):
        x_test[i,:,:,0] = cv2.imread(test_dir+'/Test/galaxy_'+
                                     str(test_ids[i])+'.png',0)/255.
    
    # now we recalculate x_pred and c_v
    z_mu, z_ls, z, y = encoder.predict(x_test, batch_size=100, verbose=True)
    #c_v = np.argmax(y, axis=1)
    labels_r = [0, 1, 2, 3, 4]
    c_v = np.argmax(y[:,(2,3,7,8,10)], axis=1)
    y = y[:,(2,3,7,8,10)]

    labels, counts = np.unique(c_v, return_counts=True)
    x_pred = VADE_gpu.predict(x_test, batch_size=200, verbose=True)
    
    features = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                          gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                          gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                          (gz_test['t01_smooth_or_features_a02_features_or_disk_debiased']>0.430))
    smooth = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                          gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                          gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                          (gz_test['t01_smooth_or_features_a01_smooth_debiased']>0.430))
    artifact = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                          gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                          gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                          (gz_test['t01_smooth_or_features_a03_star_or_artifact_debiased']>0.430))
                          
    fig = plt.figure()
    ax1 = plt.subplot2grid((len(labels)+1, len(labels)), (0, 0), colspan=len(labels), rowspan=len(labels)-1)
    ax1.hist(c_v, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], histtype='step', lw=3, label='Test Galaxies')
    ax1.hist(c_v[features[0][0:-4]], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], histtype='step', lw=3, label='GZ2-Featured Galaxies')
    ax1.hist(c_v[smooth[0][0:-4]], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], histtype='step', lw=3, label='GZ2-Smooth Galaxies')
    ax1.legend()
    ax1.set_xlabel('AstroVaDEr Component')
    ax1.set_ylabel('Count')
    
    for i in range(len(labels)):
        ax = plt.subplot2grid((len(labels)+1,len(labels)), (len(labels),i))
        img = x_pred[np.argsort(-y[:,labels[i]])][0]
        ax.imshow(img[:,:,0], cmap='gray')
        ax.axis('off')
        
    
    bar = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                      gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                      gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                     (gz_test['t01_smooth_or_features_a02_features_or_disk_debiased']>0.430)&
                    ((gz_test['t02_edgeon_a05_no_count']+gz_test['t02_edgeon_a04_yes_count'])>20)&
                     (gz_test['t02_edgeon_a05_no_debiased']>0.715)&
                    ((gz_test['t03_bar_a06_bar_count']+gz_test['t03_bar_a07_no_bar_count']>20))&
                     (gz_test['t03_bar_a06_bar_debiased']>0.8),
                     )
            
    nobar = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                      gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                      gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                     (gz_test['t01_smooth_or_features_a02_features_or_disk_debiased']>0.430)&
                    ((gz_test['t02_edgeon_a05_no_count']+gz_test['t02_edgeon_a04_yes_count'])>20)&
                     (gz_test['t02_edgeon_a05_no_debiased']>0.715)&
                    ((gz_test['t03_bar_a06_bar_count']+gz_test['t03_bar_a07_no_bar_count']>20))&
                     (gz_test['t03_bar_a07_no_bar_debiased']>0.8),
                     )
            
    fig = plt.figure()
    ax1 = plt.subplot2grid((len(labels)+1, len(labels)), (0, 0), colspan=len(labels), rowspan=len(labels)-1)
    ax1.hist(c_v[bar[0]], histtype='step', lw=3,
             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
                   label='GZ2-Barred Spirals, N={0}'.format(bar[0].shape[0]))
    ax1.hist(c_v[nobar[0][0:-1]], histtype='step', lw=3,
             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
                   label='GZ2-Unbarred Spirals, N={0}'.format(nobar[0].shape[0]))
    ax1.legend(loc=2)
    ax1.set_xlabel('AstroVaDEr Component')
    ax1.set_ylabel('Fraction')
    ax1.set_xlim(-0.5,11.5)
    
    for i in range(len(labels)):
        ax = plt.subplot2grid((len(labels)+1,len(labels)), (len(labels),i))
        img = x_pred[np.argsort(-y[:,labels[i]])][0]
        ax.imshow(img[:,:,0], cmap='gray')
        ax.axis('off')
        
    plt.subplots_adjust(top=0.99,bottom=0.01,left=0.11,right=0.99,hspace=0.0,
                        wspace=0.0)
    
    plt.show()
    
    edge = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                       gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                       gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                      (gz_test['t01_smooth_or_features_a02_features_or_disk_debiased']>0.430)&
                     ((gz_test['t02_edgeon_a05_no_count']+gz_test['t02_edgeon_a04_yes_count'])>20)&
                      (gz_test['t02_edgeon_a04_yes_debiased']>0.715)
                     )
            
    noedge = np.where( ((gz_test['t01_smooth_or_features_a03_star_or_artifact_count']+
                         gz_test['t01_smooth_or_features_a02_features_or_disk_count']+
                         gz_test['t01_smooth_or_features_a01_smooth_count']) > 20)&
                        (gz_test['t01_smooth_or_features_a02_features_or_disk_debiased']>0.430)&
                       ((gz_test['t02_edgeon_a05_no_count']+gz_test['t02_edgeon_a04_yes_count'])>20)&
                        (gz_test['t02_edgeon_a05_no_debiased']>0.715)
                     )
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((len(labels)+1, len(labels)), (0, 0), colspan=len(labels), rowspan=len(labels)-1)
    ax1.hist(c_v[edge[0]], histtype='step', lw=3,
             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
                   label='GZ2-Edge On Spirals, N={0}'.format(edge[0].shape[0]))
    ax1.hist(c_v[noedge[0][0:-1]], histtype='step', lw=3,
             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
                   label='GZ2-Face On Spirals, N={0}'.format(noedge[0].shape[0]))
    ax1.legend(loc=1)
    ax1.set_xlabel('AstroVaDEr Component')
    ax1.set_ylabel('Fraction')
    ax1.set_xlim(-0.5,11.5)
    
    for i in range(12):
        ax = plt.subplot2grid((13,12), (12,i))
        img = x_pred[np.argsort(-y[:,labels[i]])][0]
        ax.imshow(img[:,:,0], cmap='gray')
        ax.axis('off')
        
    plt.subplots_adjust(top=0.99,bottom=0.01,left=0.11,right=0.99,hspace=0.0,
                        wspace=0.0)
    
    plt.show()
    
    ring = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a19_ring_debiased']>0.4)&
                      (gz_test['t08_odd_feature_a19_ring_count']>5)
                     )
    lens = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a20_lens_or_arc_debiased']>0.4)&
                      (gz_test['t08_odd_feature_a20_lens_or_arc_count']>5)
                     )
    disturbed = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a21_disturbed_debiased']>0.4)&
                      (gz_test['t08_odd_feature_a21_disturbed_count']>5)
                     )
    ireg = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a22_irregular_debiased']>0.4)&
                      (gz_test['t08_odd_feature_a22_irregular_count']>5)
                     )
    other = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a23_other_debiased']>0.4)&
                      (gz_test['t08_odd_feature_a23_other_count']>5)
                     )
    merger = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a24_merger_debiased']>0.4)&
                      (gz_test['t08_odd_feature_a24_merger_count']>5)
                     )
    dust = np.where( ((gz_test['t06_odd_a14_yes_count']+
                      gz_test['t06_odd_a15_no_count']) > 20)&
                     (gz_test['t08_odd_feature_a38_dust_lane_debiased']>0.4)&
                     (gz_test['t08_odd_feature_a38_dust_lane_count']>5)
                     )
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((len(labels)+1, len(labels)), (0, 0), colspan=len(labels), rowspan=len(labels)-1)
#    ax1.hist(c_v[ring[0]], histtype='step', lw=3,
#             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
#                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
#                   label='GZ2-Ring, N={0}'.format(ring[0].shape[0]))
    ax1.hist(c_v[lens[0][0:-1]], histtype='step', lw=3,
             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=False,
                   label='GZ2-Lens Or Arc, N={0}'.format(lens[0].shape[0]))
#    ax1.hist(c_v[disturbed[0][0:-1]], histtype='step', lw=3,
#             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
#                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
#                   label='GZ2-Disturbed, N={0}'.format(disturbed[0].shape[0]))
#    ax1.hist(c_v[ireg[0][0:-1]], histtype='step', lw=3,
#             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
#                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
#                   label='GZ2-Irregular, N={0}'.format(ireg[0].shape[0]))
#    ax1.hist(c_v[merger[0][0:-1]], histtype='step', lw=3,
#             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
#                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
#                   label='GZ2-Merger, N={0}'.format(merger[0].shape[0]))
#    ax1.hist(c_v[dust[0][0:-1]], histtype='step', lw=3,
#             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
#                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
#                   label='GZ2-Dustlane, N={0}'.format(dust[0].shape[0]))
#    ax1.hist(c_v[other[0][0:-1]], histtype='step', lw=3,
#             bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
#                   6.5, 7.5, 8.5, 9.5, 10.5, 11.5], density=True,
#                   label='GZ2-Other, N={0}'.format(other[0].shape[0]))
    ax1.legend(loc=0)
    ax1.set_xlabel('AstroVaDEr Component')
    ax1.set_ylabel('Fraction')
    ax1.set_xlim(-0.5,11.5)
    
    for i in range(len(labels)):
        ax = plt.subplot2grid((len(labels)+1,len(labels)), (len(labels),i))
        img = x_test[np.argsort(-y[lens[0][0:-1]][:,labels[i]])][0]
        ax.imshow(img[:,:,0], cmap='gray')
        ax.axis('off')
        
    plt.subplots_adjust(top=0.99,bottom=0.01,left=0.11,right=0.99,hspace=0.0,
                        wspace=0.0)
    
    plt.show()
