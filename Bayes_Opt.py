"""
Bayesian Hyperparameter Optimisation using hyperopt
   - sample hyperparamters from a prior dist, learn best
     inputs by modifying the dist based on the resulting
     loss
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import numpy as np
import csv
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D, Conv2D, Lambda
import keras.backend as K
from keras import Model, optimizers, regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler
#from keras.utils import multi_gpu_model

from hyperas import optim
from hyperas.distributions import uniform, quniform, loguniform, choice
from hyperopt import Trials, STATUS_OK, tpe

from s3VDC import VAE_sampling, cond_z, gamma_training,\
                    KL_metric, recon_metric, lr_schedule
                    

def data():

    datagen = ImageDataGenerator(rescale=1./255,
                             horizontal_flip=True, vertical_flip=True,
                             dtype='float32', fill_mode='wrap',
                             preprocessing_function=None,)
    train_generator = datagen.flow_from_directory('/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Images/Train', target_size=(128,128),
                                              color_mode='grayscale', class_mode='input',
                                              batch_size=100,
                                              shuffle=True)
    valid_generator = datagen.flow_from_directory('/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Images/Train', target_size=(128,128),
                                              color_mode='grayscale', class_mode='input',
                                              batch_size=2500,
                                              shuffle=True)
    x_test, y_test = valid_generator.next()
    del valid_generator
    return train_generator, x_test, y_test
    
def VAE_Model(train_generator, x_test, y_test):
    
    f1 = {{choice([128,64,32])}}
    f2 = {{choice([64, 32])}}
    f3 = {{choice([32, 16])}}
    filters = [f1, f2, f3] #from Walmsley et al 2019
    k1 = {{choice([3,5])}}
    k2 = {{choice([3,5])}}
    k3 = {{choice([3,5])}}
    kernels = [(k1,k1), (k2,k2), (k3,k3)]
    flat_units = [f3*16*16]
    In_shape = (128,128,1)
    
    latents = int({{quniform(8,128,1)}})
    
    #%%============================================================================
    # Encoder Setup
    I = Input(shape=In_shape, dtype='float32', name='encoder_input') # (?, 128, 128, b)
    
    Conv1 = Conv2D(filters[0], kernel_size=kernels[0], activation='relu',
                   padding='same', name='Conv1',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(I) # (?, 128, 128, 32)
    Conv2 = Conv2D(filters[0], kernel_size=kernels[0], activation='relu',
                   padding='same', name='Conv2',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Conv1) # (?, 128, 128, 32)
    Pool1 = MaxPooling2D((2,2), name='Pool1')(Conv2) # (?, 64, 64, 32)
    
    Conv3 = Conv2D(filters[1], kernel_size=kernels[1], activation='relu',
                   padding='same', name='Conv3',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Pool1) # (?, 64, 64, 32)
    Conv4 = Conv2D(filters[1], kernel_size=kernels[1], activation='relu',
                   padding='same', name='Conv4',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Conv3) # (?, 64, 64, 32)
    Pool2 = MaxPooling2D((2,2), name='Pool2')(Conv4) # (?, 32, 32, 32)
    
    Conv5 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv5',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Pool2) # (?, 32, 32, 16)
    Conv6 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv6',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Conv5) # (?, 32, 32, 16)
    Pool3 = MaxPooling2D((2,2), name='Pool3')(Conv6) # (?, 16, 16, 16)
    
    
    Flat = Flatten(name='Flat')(Pool3) # (?, 4096)
    FC1 = Dense(128, name='FC1', activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Flat)
    
    #%%============================================================================
    # Embedding Layer
    
    z_mean = Dense(latents, name='latentmean',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(FC1) # Edit namespace for CAE
    z_log_var = Dense(latents, name='latentlog_var',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(FC1) # Comment out for CAE
        
    z = Lambda(VAE_sampling, output_shape=(latents,),
               name='latentz_sampling')([z_mean,z_log_var]) # repara trick
    
    y_training = K.variable(True, dtype='bool')
    z_out = Lambda(cond_z(y_training), output_shape=(latents,),
                   name='conditional_z_out')([z_mean, z]) # set input for decoder
    
    #%%============================================================================
    # Decoder Setup
    
    dec_in = Input(shape=(latents,), dtype='float32', name='decoder_input')
    FC2 = Dense(128, activation='relu', name='FC2',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(dec_in)
    FC3 = Dense(flat_units[0], activation='relu', name='FC3',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(FC2) # (?, 4096)
    reshape = Reshape((16,16,filters[2]), name='reshape')(FC3) # (?, 16, 16, 16)
    
    Conv7 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv7',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(reshape) # (?, 16, 16, 16)
    Conv8 = Conv2D(filters[2], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv8',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Conv7) # (?, 16, 16, 16)
    Up1 = UpSampling2D((2,2), interpolation='nearest',
                       name='Up1')(Conv8) # (?, 32, 32, 16)
    
    Conv9 = Conv2D(filters[1], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv9',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Up1) # (?, 32, 32, 32)
    Conv10 = Conv2D(filters[1], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv10',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Conv9) # (?, 32, 32, 32)
    Up2 = UpSampling2D((2,2), interpolation='nearest',
                       name='Up2')(Conv10) # (?, 64, 64, 32)
    
    Conv11 = Conv2D(filters[0], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv11',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Up2) # (?, 64, 64, 32)
    Conv12 = Conv2D(filters[0], kernel_size=kernels[2], activation='relu',
                   padding='same', name='Conv12',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Conv11) # (?, 64, 64, 32)
    Up3 = UpSampling2D((2,2), interpolation='nearest',
                       name='Up3')(Conv12) # (?, 128, 128, 32)
    
    Out = Conv2D(1, kernel_size=kernels[0], activation='relu',
                 padding='same', name='Output',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(Up3) # (?, 128, 128, b)
    
    outputs = { 'z' : z,
                'z_mean' : z_mean,
                'z_log_var' : z_log_var,
                'z_out' : z_out,
                'decoder_out' : Out}

    #%%============================================================================
    # Model Setup
        
    dec = Model(inputs=dec_in, outputs=outputs['decoder_out'], name='Decoder')
    
    VAE = Model(inputs=I, outputs=dec(outputs['z_out']), name='VAE')
        
    # Losses and Metrics
    
    beta = {{choice([5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])}}    
    params = {'original_dims' : 128*128,
              'latents' : latents,
              'loss_type' : 'mse',
              'warm_up' : beta,}
    warm_up_loss = gamma_training(outputs, params)
    vae_loss = KL_metric(outputs)
    recon_loss = recon_metric(params)
    
    lr = {{loguniform(-10,-6.9)}}
    lr_decay = {{uniform(0.9,0.99)}}
    lr_steps = int({{quniform(1,5,1)}})

    lr_decayCB = LearningRateScheduler(lr_schedule(lr, 1e-6,
                                                   lr_steps, lr_decay),
                                       )
    es = EarlyStopping(monitor='loss', mode='min', patience=2,
                       min_delta=0.05, )

    #VAE_gpu = multi_gpu_model(VAE, gpus=2)

    VAE.compile(optimizers.Adam(lr=lr),
                    loss = warm_up_loss,
                    metrics = [recon_loss,vae_loss])
    
    start = time.time()
    VAE.fit_generator(train_generator, epochs=10,
                              steps_per_epoch=1596, initial_epoch=0,
                              workers=32, max_queue_size=500,
                              callbacks=[lr_decayCB, es],
                       )

    valid_loss = VAE.evaluate(x_test, y_test, batch_size=100, verbose=1)
    end = time.time()
    
    # Write to the csv file ('a' means append)
    out_file = '/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/Bayes_Opt_Runs.csv'
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([valid_loss[0], valid_loss[1], valid_loss[2],
                     STATUS_OK, end-start, f1, f2, f3, k1, k2, k3,
                     latents, beta, lr, lr_decay, lr_steps])
    of_connection.close()
    
    return {'loss': valid_loss[0], 'recon': valid_loss[1], 'kl': valid_loss[2],
            'status': STATUS_OK, 'model': VAE}
    
if __name__ == '__main__':
    bayes_trials = Trials()
    # File to save first results (tail .csv lets us check file while running)
    out_file = '/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/Bayes_Opt_Runs.csv'
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    # Write the headers to the file
    writer.writerow(['Network_Loss', 'Recon_Loss', 'KL_loss', 'Status', 'Time',
                     'Filter 1', 'Filter 2', 'Filter 3', 'Kernel 1',
                     'Kernel 2', 'Kernel 3', 'Latents', 'Beta', 'LR',
                     'LR decay', 'LR Steps'])
    of_connection.close()
    best_run, best_model, space = optim.minimize(
                                  model=VAE_Model,
                                  data=data,
                                  algo=tpe.suggest,
                                  max_evals=64,
                                  trials=bayes_trials,
                                  return_space=True,
                                  )  
    #import pickle
    #pickle.dump(bayes_trials, open('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/bayes_Trials_database.p', 'wb'))
    #pickle.dump(space, open('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/bayes_opt_space.p', 'wb'))
    for trial in bayes_trials:
        print(trial)
    for sp in space:
        print(sp)
    best_model.save_weights('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/Bayes_Opt_gammatraining_bestweights.h5')