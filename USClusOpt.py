#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:23:02 2020

@author: ashleyspindler

Unsupervised clustering optimisation task. Perform y-training strategy
on dataset of 20,000 inputs. Train GMM with variety of number components
and record clustering metrics and log(p).
"""

import configparser #used to read config file
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape, MaxPooling2D,\
                         UpSampling2D, Conv2D, Lambda, GaussianNoise,\
                         LeakyReLU
import keras.backend as K
from keras import Model, optimizers, regularizers
from keras.callbacks import LearningRateScheduler

import sklearn.cluster as cluster
import sklearn.mixture as mixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from AE_GMM_Layer import GMMLayer_2
from AV_Losses import gamma_training, beta_training, static_training, KL_metric, recon_metric, VaDE_metric, log_prob
from AV_Layers import VAE_sampling, get_gamma, cond_z
from AV_Callbacks import lr_schedule, AnnealingCallback
from clustering_metrics_numpy import metric_simplified_silhouette, metric_calinski_harabaz_numpy, metric_cluster_separation

def input_to_flat(In_shape, params, filters, kernels):
    I = Input(shape=In_shape, dtype='float32', name='encoder_input') # (?, 128, 128, b)
    Noise = GaussianNoise(1e-8)(I)
    
    Conv1 = Conv2D(filters[0], kernel_size=kernels[0],
                   padding='same', name='Conv1',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Noise) # (?, 128, 128, 32)
    Conv1 = LeakyReLU(alpha=0.1)(Conv1)
    Conv2 = Conv2D(filters[0], kernel_size=kernels[0],
                   padding='same', name='Conv2',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv1) # (?, 128, 128, 32)
    Conv2 = LeakyReLU(alpha=0.1)(Conv2)
    Pool1 = MaxPooling2D((2,2), name='Pool1')(Conv2) # (?, 64, 64, 32)
    
    Conv3 = Conv2D(filters[1], kernel_size=kernels[1],
                   padding='same', name='Conv3',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Pool1) # (?, 64, 64, 32)
    Conv3 = LeakyReLU(alpha=0.1)(Conv3)
    Conv4 = Conv2D(filters[1], kernel_size=kernels[1],
                   padding='same', name='Conv4',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv3) # (?, 64, 64, 32)
    Conv4 = LeakyReLU(alpha=0.1)(Conv4)
    Pool2 = MaxPooling2D((2,2), name='Pool2')(Conv4) # (?, 32, 32, 32)
    
    Conv5 = Conv2D(filters[2], kernel_size=kernels[2],
                   padding='same', name='Conv5',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Pool2) # (?, 32, 32, 16)
    Conv5 = LeakyReLU(alpha=0.1)(Conv5)
    Conv6 = Conv2D(filters[2], kernel_size=kernels[2],
                   padding='same', name='Conv6',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv5) # (?, 32, 32, 16)
    Conv6 = LeakyReLU(alpha=0.1)(Conv6)
    Pool3 = MaxPooling2D((2,2), name='Pool3')(Conv6) # (?, 16, 16, 16)
    
    
    Flat = Flatten(name='Flat')(Pool3) # (?, 4096)
    
    return I, Flat

def enc_to_embedded(In, params, y_training):
    z_mean = Dense(params['latents'], name='latentmean',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(In) # Edit namespace for CAE
    z_log_var = Dense(params['latents'], name='latentlog_var',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(In) # Comment out for CAE
    
    z = Lambda(VAE_sampling, output_shape=(params['latents'],),
               name='latentz_sampling')([z_mean,z_log_var]) # repara trick
    
    z_out = Lambda(cond_z(y_training), output_shape=(params['latents'],),
               name='conditional_z_out')([z_mean, z]) # set input for decoder
    
    return z_mean, z_log_var, z, z_out

def embedded_to_gmm(params, z_out, y_training):
    GMM = GMMLayer_2(params['latents'], params['clusters'], name='latentGMM_Layer')
    z_gmm = GMM(z_out) # pass through layer containing GMM weights
    
    gmm_weights = { 'theta' : GMM.weights[0],
                    'mu' : GMM.weights[1],
                    'lambda' : GMM.weights[2],}
    gamma_out, z_gmm = get_gamma(gmm_weights, params)(z_gmm)
    
    return z_gmm, gamma_out, gmm_weights
    
if __name__ == "__main__":
    #%%============================================================================
    # Initial Setup
    
    """ Hyperparameters
    TODO
    """
    
    print('==================== Loading Hyperparameters ====================')
    print('=================================================================')
    
    config = configparser.ConfigParser()
    config.read('s3VDC-optconfig.txt') 
    
    Model_Name = config['directories']['Model_Name']
    train_dir = config['directories']['train_dir']
    model_dir = config['directories']['model_dir']
    filters = [64,64,16]
    kernels = [(3,3), (5,5), (5,5)]
    flat_units = [4096]
    
    params = {}
    
    params['batch_size'] = int(config['training']['batch_size'])
    params['gmm_steps'] = int(config['training']['gmm_steps'])
    params['warm_up_steps'] = 25
    params['annealing_steps'] = 10
    params['static_steps'] = 30
    params['annealing_periods'] = 2
    params['total_steps'] = params['warm_up_steps'] + params['annealing_periods']\
                           * (params['annealing_steps'] + params['static_steps'])
    params['l2_regularizer'] = float(config['training']['l2_regulariser'])
    
    
    side = int(config['dataset']['side'])
    b = int(config['dataset']['b'])
    cm = 'grayscale' if b==1 else 'rgb'
    In_shape = (side,side,b)
    params['original_dims'] = side*side*b
    
    params['latents'] = 20#int(config['embedding']['latents'])
    params['clusters'] = int(config['embedding']['clusters'])
    
    params['lr'] = 0.0003#float(config['lr_settings']['lr'])
    params['min_lr'] = 1e-5#float(config['lr_settings']['min_lr'])
    params['lr_steps'] = 3#int(config['lr_settings']['lr_steps'])
    params['lr_decay'] = np.power(params['min_lr']/params['lr'],
                                  params['lr_steps']/params['total_steps'])
    
    params['warm_up'] = float(config['loss_settings']['warm_up_factor'])
    params['annealing_factor'] = K.variable(float(config['loss_settings']['warm_up_factor']), dtype='float32')
    params['loss_type'] = config['loss_settings']['loss_type']
    params['output_activity'] = 'sigmoid' if params['loss_type']=='binary_crossentropy' else 'relu'
    
    #%%============================================================================
    # Create image data generator
    print('==================== Loading Training Data ====================')
    print('===============================================================')
    
    
    datagen = ImageDataGenerator(rescale=1./255, zoom_range=(0.75,0.75),
                                 dtype='float32', fill_mode='wrap')
    train_generator = datagen.flow_from_directory(train_dir, target_size=(side,side),
                                                  color_mode=cm, class_mode='input',
                                                  batch_size=params['batch_size'],
                                                  shuffle=True)
    
    x_train, _ = train_generator.next()
    for i in range(0, params['gmm_steps']-1):
        data, _ = train_generator.next()
        x_train = np.append(x_train, data, axis=0)
    
    #%%============================================================================
    # Encoder Setup
    
    print('==================== Constructing Model ====================')
    print('============================================================')
    
    enc_in, enc_flat = input_to_flat(In_shape, params, filters, kernels)
    
    #%%============================================================================
    # Decoder Setup
    
    dec_in = Input(shape=(params['latents'],), dtype='float32', name='decoder_input')
    FC2 = Dense(flat_units[0], name='FC3',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(dec_in) # (?, 4096)
    FC2 = LeakyReLU(alpha=0.1)(FC2)
    reshape = Reshape((16,16,filters[2]), name='reshape')(FC2) # (?, 16, 16, 16)
    
    Conv7 = Conv2D(filters[2], kernel_size=kernels[2],
                   padding='same', name='Conv7',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(reshape) # (?, 16, 16, 16)
    Conv7 = LeakyReLU(alpha=0.1)(Conv7)
    Conv8 = Conv2D(filters[2], kernel_size=kernels[2],
                   padding='same', name='Conv8',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv7) # (?, 16, 16, 16)
    Conv8 = LeakyReLU(alpha=0.1)(Conv8)
    Up1 = UpSampling2D((2,2), interpolation='nearest',
                       name='Up1')(Conv8) # (?, 32, 32, 16)
    
    Conv9 = Conv2D(filters[1], kernel_size=kernels[2],
                   padding='same', name='Conv9',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Up1) # (?, 32, 32, 32)
    Conv9 = LeakyReLU(alpha=0.1)(Conv9)
    Conv10 = Conv2D(filters[1], kernel_size=kernels[2],
                   padding='same', name='Conv10',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv9) # (?, 32, 32, 32)
    Conv10 = LeakyReLU(alpha=0.1)(Conv10)
    Up2 = UpSampling2D((2,2), interpolation='nearest',
                       name='Up2')(Conv10) # (?, 64, 64, 32)
    
    Conv11 = Conv2D(filters[0], kernel_size=kernels[2],
                   padding='same', name='Conv11',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Up2) # (?, 64, 64, 32)
    Conv11 = LeakyReLU(alpha=0.1)(Conv11)
    Conv12 = Conv2D(filters[0], kernel_size=kernels[2],
                   padding='same', name='Conv12',
                   kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                   bias_regularizer=regularizers.l2(params['l2_regularizer']),
                   )(Conv11) # (?, 64, 64, 32)
    Conv12 = LeakyReLU(alpha=0.1)(Conv12)
    Up3 = UpSampling2D((2,2), interpolation='nearest',
                       name='Up3')(Conv12) # (?, 128, 128, 32)
    
    Out = Conv2D(b, kernel_size=kernels[0], activation=params['output_activity'],
                 padding='same', name='Output',
                 kernel_regularizer=regularizers.l2(params['l2_regularizer']),
                 bias_regularizer=regularizers.l2(params['l2_regularizer']),
                 )(Up3) # (?, 128, 128, b)
    
    #%%============================================================================
    # initial y-training
    print('================ Performing Initial y-Training ================')
    print('===============================================================')
    
    
    y_training = K.variable(False, dtype='bool')
    z_mean, z_log_var, z, z_out = enc_to_embedded(enc_flat, params, y_training)
    
    outputs = { 'z' : z,
                'z_mean' : z_mean,
                'z_log_var' : z_log_var,
                'z_out' : z_out,
                'decoder_out' : Out}
    
    enc_y = Model(inputs=enc_in, outputs=z_out)
    dec = Model(inputs=dec_in, outputs=Out)
    
    VAE_y = Model(inputs=enc_in, outputs=dec(z_out))
    
    warm_up_loss = gamma_training(outputs, params)
    vae_loss = KL_metric(outputs)
    recon_loss = recon_metric(params)
    lr_decayCB = LearningRateScheduler(lr_schedule(params['lr'], params['min_lr'], params['lr_steps'], params['lr_decay']), verbose=1)
    
    VAE_y.compile(optimizers.Adam(lr=params['lr']),
                loss = warm_up_loss,
                metrics = [recon_loss,vae_loss])
    
    VAE_y.fit(x_train, x_train, epochs=25,
                        batch_size=100, initial_epoch=0,
                        callbacks=[lr_decayCB],
                        )
    
    VAE_y.save_weights(model_dir+'ClusteringOpt/Model_weights_pretrain.hdf5')
    
    x_pred = VAE_y.predict(x_train[0:100])
    f, ax = plt.subplots(10,10)
    [[ax.ravel()[j].imshow(x_pred[j,:,:,0], vmin=0, vmax=1, cmap='binary') for j in range(100)]]
    plt.show()
    plt.close()
    
    #%%============================================================================
    # Make a GMM_layer with increasing cluster numbers, train for 2.5 periods
    # of 10 Beta and 5 Static epochs
    
    print('================ Mini-batch GMM initialization ================')
    print('===============================================================')
    print('Collecting {0} mini-batches for GMM fit'.format(params['gmm_steps']))
    
    gmm_data = enc_y.predict(x_train, verbose=True)
    
    P = PCA(n_components=2)
    p = P.fit_transform(gmm_data)
    plt.plot(p[:,0], p[:,1], '.')
    plt.show()
    
    print('Mini-batches collected')
    
    print('================== Starting Cluster Opt Loop ==================')
    print('===============================================================')
    
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64]
    lines = ['k', 'k:', 'r', 'r:', 'y', 'y:', 'b', 'b:', 'c', 'c:']
    ch_met = np.zeros(len(n_clusters))
    ss_met = np.zeros(len(n_clusters))
    cs_met = np.zeros(len(n_clusters))
    sl_met = np.zeros(len(n_clusters))
    bl_met = np.zeros(len(n_clusters))
    rl_met = np.zeros(len(n_clusters))
    kl_met = np.zeros(len(n_clusters))
    lp_met = np.zeros(len(n_clusters))
    
    f, ax = plt.subplots(2,5)
    a = ax.ravel()
    h, cx = plt.subplots(2,2)
    cx = cx.ravel()
    
    for i in range(len(n_clusters)):
        print('==================== Starting gmm training ====================')
        print('===============================================================')
        # make gmm layer
        params['clusters'] = n_clusters[i]
        print('Training for s3VDC Model with {0} clusers'.
              format(params['clusters']))
        
        z_gmm, gamma_out, gmm_weights = embedded_to_gmm(params, z_out,
                                                        y_training)
        outputs['gamma'] = gamma_out
        outputs['z_gmm'] = z_gmm
        
        enc_gmm = Model(inputs=enc_in, outputs=z_out) #new encoder
        enc_gamma = Model(inputs=enc_in, outputs=gamma_out) #gamma encoder
        VAE_gmm = Model(inputs=enc_in, outputs=(dec(z_gmm))) #new model
        VAE_gmm.load_weights(model_dir+'ClusteringOpt/Model_weights_pretrain.hdf5',
                             by_name = True) #load y_trained weights
        
        beta_loss = beta_training(outputs, gmm_weights, params)
        static_loss = static_training(outputs, gmm_weights, params)
        vae_loss = KL_metric(outputs)
        recon_loss = recon_metric(params)
        vade_loss = VaDE_metric(outputs, gmm_weights, params)
        log_prob_met = log_prob(z_gmm, gmm_weights, params)
        
        kmeans = cluster.KMeans(n_clusters=params['clusters'])
        kmeans.fit(gmm_data)
        
        print('\n')
        print('Performing Gaussian Mixture Modelling to find cluster covariances')
        
        gmm = mixture.GaussianMixture(n_components=params['clusters'],
                                      covariance_type='diag',
                                      max_iter=10000,
                                      means_init=kmeans.cluster_centers_,
                                      verbose=1)
        gmm.fit(gmm_data)
        
        mtm = (1-params['warm_up'])**3 # momentum factor to update GMM layer weights
        
        print('\n')
        print('Updating GMM weights in Model with momentum {0:.3e}'.format(mtm))
        
        K.set_value(gmm_weights['theta'], (1.0 - mtm) * K.get_value(gmm_weights["theta"]) + mtm * gmm.weights_.T)
        K.set_value(gmm_weights['mu'], (1.0 - mtm) * K.get_value(gmm_weights["mu"]) + mtm * gmm.means_.T)
        K.set_value(gmm_weights['lambda'], (1.0 - mtm) * K.get_value(gmm_weights["lambda"]) + mtm * gmm.covariances_.T)
        
        K.set_value(y_training, False)
        lr_new = params['lr'] * np.power(params['lr_decay'],
                         np.floor(params['warm_up_steps']/params['lr_steps']))
                
        total_epochs = params['total_steps']
        
        AnnealingCB = AnnealingCallback(params)
        
        current_epoch = params['warm_up_steps']+1
        while current_epoch < total_epochs:
            # set up beta annealing loss training
            VAE_gmm.compile(optimizers.Adam(lr=lr_new, clipvalue=1),
                loss = beta_loss,
                metrics = [beta_loss,static_loss,recon_loss,
                           vae_loss,vade_loss,log_prob_met])
            
            VAE_gmm.fit(x_train, x_train, epochs=params['annealing_steps']+current_epoch,
                                  batch_size=100, initial_epoch=current_epoch,
                                  callbacks=[AnnealingCB,lr_decayCB],
                           )
            
            xlabels = np.arange(current_epoch, current_epoch+params['annealing_steps'])
            
            cx[0].plot(xlabels, VAE_gmm.history.history['static_loss'],
                   #label='N clusters = {0}'.format(params['clusters']),
                   lines[i])
            
            cx[1].plot(xlabels, VAE_gmm.history.history['beta_loss'], lines[i])
            
            cx[2].plot(xlabels, VAE_gmm.history.history['recon_loss'], lines[i])
            
            cx[3].plot(xlabels, VAE_gmm.history.history['kl_loss'], lines[i])
            
            current_epoch += params['annealing_steps']
            
            # set up static fine tuning
            VAE_gmm.compile(optimizers.Adam(lr=lr_new, clipvalue=1),
                loss = static_loss,
                metrics = [beta_loss,static_loss,recon_loss,
                           vae_loss,vade_loss,log_prob_met])
            
            VAE_gmm.fit(x_train, x_train, epochs=params['static_steps']+current_epoch,
                                  batch_size=100, initial_epoch=current_epoch,
                                  callbacks=[AnnealingCB,lr_decayCB],
                           )
            
            xlabels = np.arange(current_epoch, current_epoch+params['static_steps'])
            
            cx[0].plot(VAE_gmm.history.history['static_loss'],
                   #label='N clusters = {0}'.format(params['clusters']),
                   lines[i])
            
            cx[1].plot(VAE_gmm.history.history['beta_loss'], lines[i])
            
            cx[2].plot(VAE_gmm.history.history['recon_loss'], lines[i])
            
            cx[3].plot(VAE_gmm.history.history['kl_loss'], lines[i])
            
            current_epoch += params['static_steps']
        
        print('==================== Collecting Metrics  ====================')
        print('=============================================================')
        _, bl_met[i], sl_met[i], rl_met[i], kl_met[i], _, lp_met[i]\
            = VAE_gmm.evaluate(x_train, x_train,
                               batch_size=100,
                               verbose=1)
        z_pred = enc_gmm.predict(x_train, batch_size=100, verbose=1)
        y = enc_gamma.predict(x_train, batch_size=100, verbose=1)
        c = np.argmax(y,axis=1)
        ch_met[i] = metric_calinski_harabaz_numpy(z_pred, c,
                                                  params['clusters'])
        ss_met[i] = metric_simplified_silhouette(K.get_value(gmm_weights['mu']).T,
                                              z_pred, c)
        cs_met[i] = metric_cluster_separation(K.get_value(
                                                gmm_weights['mu']),
                                                num_clusters=params['clusters'])
        
        print('================== Performing TSNE Embedding ==================')
        print('===============================================================')
        t = TSNE(n_components=2, verbose=2, perplexity=30, early_exaggeration=5,
                 learning_rate=1000, n_iter=5000, n_jobs=-1)
        t_idx = np.random.choice(20000, 5000)
        t.fit(z_pred[t_idx]/1000)
        
        print('==================== Plotting and Saving ====================')
        print('=============================================================')
        
        a[i].scatter(x=t.embedding_[:,0], y=t.embedding_[:,1], c=c[t_idx],
                  marker='o', s=1, edgecolor=None)
        a[i].set_title('N Clusters = {0}'.format(params['clusters']))
                
        # Save some stuff
        np.save(model_dir+'ClusteringOpt/z_prediction_{0}.npy'.
                format(params['clusters']), z_pred)
        np.save(model_dir+'ClusteringOpt/gamma_prediction_{0}.npy'.
                format(params['clusters']), y)
        np.save(model_dir+'ClusteringOpt/cluster_prediction_{0}.npy'.
                format(params['clusters']), c)
        np.save(model_dir+'ClusteringOpt/tsne_embedding_{0}.npy'.
                format(params['clusters']), t.embedding_)
        np.save(model_dir+'ClusteringOpt/static_losses_{0}.npy'.
                format(params['clusters']), VAE_gmm.history.history['static_loss'])
        np.save(model_dir+'ClusteringOpt/beta_losses_{0}.npy'.
                format(params['clusters']), VAE_gmm.history.history['beta_loss'])
        np.save(model_dir+'ClusteringOpt/recon_losses_{0}.npy'.
                format(params['clusters']), VAE_gmm.history.history['recon_loss'])
        np.save(model_dir+'ClusteringOpt/kl_losses_{0}.npy'.
                format(params['clusters']), VAE_gmm.history.history['kl_loss'])
        VAE_gmm.save_weights(model_dir+'ClusteringOpt/Model_weights_{0}.hdf5'.
                format(params['clusters']))

# finish up

print('==================== Final Plots and Files ====================')
print('===============================================================')

g, bx = plt.subplots(2,2)
labels = [[str(n) for n in n_clusters]][0]
bx[0,0].plot(labels, ch_met, lw=2, color='k')
bx[0,0].set_ylabel('Calinski-Harabaz Score', fontsize='small')
bx[0,0].set_xlabel('N Clusters', fontsize='small')
bx[0,1].plot(labels, ss_met, lw=2, color='k')
bx[0,1].set_ylabel('Simplified Silhouette Coefficient', fontsize='small')
bx[0,1].set_xlabel('N Clusters', fontsize='small')
bx[1,0].plot(labels, cs_met, lw=2, color='k')
bx[1,0].set_ylabel('Cluster Seperation Metric', fontsize='small')
bx[1,0].set_xlabel('N Clusters', fontsize='small')
bx[1,1].plot(labels, lp_met, lw=2, color='k')
bx[1,1].set_ylabel('Mean log probability of samples', fontsize='small')
bx[1,1].set_xlabel('N Clusters', fontsize='small')

cx[0].set_xlabel('Epochs', fontsize='small')
cx[0].set_ylabel('$L_{beta}$', fontsize='small')
#cx[0].legend(fontsize='small')
cx[1].set_xlabel('Epochs', fontsize='small')
cx[1].set_ylabel('$L_{static}}$', fontsize='small')
cx[2].set_xlabel('Epochs', fontsize='small')
cx[2].set_ylabel('$L_{recon}$', fontsize='small')
cx[3].set_xlabel('Epochs', fontsize='small')
cx[3].set_ylabel('$L_{KL}$', fontsize='small')

f.savefig(model_dir+'ClusteringOpt/TSNE_embeddings.png')
h.savefig(model_dir+'ClusteringOpt/Loss_History.png')
g.savefig(model_dir+'ClusteringOpt/Clustering_metrics.png')

np.save(model_dir+'ClusteringOpt/Calinski-Harabaz.npy', ch_met)
np.save(model_dir+'ClusteringOpt/Simplified-Silhouette.npy', ss_met)
np.save(model_dir+'ClusteringOpt/Cluster-Seperation.npy', cs_met)
np.save(model_dir+'ClusteringOpt/log-probability.npy', lp_met)
np.save(model_dir+'ClusteringOpt/beta-loss.npy', bl_met)
np.save(model_dir+'ClusteringOpt/recon-loss.npy', rl_met)
np.save(model_dir+'ClusteringOpt/kl-loss.npy', kl_met)
np.save(model_dir+'ClusteringOpt/static-loss.npy', sl_met)

plt.show()
plt.close()
