#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:47:16 2020

@author: ashleyspindler

Script to import a trained encoder and decoder, embed test data and perform
bayesian mixture modelling
"""

from keras import models

import configparser
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage.exposure import rescale_intensity

from sklearn.mixture import BayesianGaussianMixture

def contrast_enhance(img):

    img_rescale = rescale_intensity(img, in_range=(30, 255), out_range=(0,255))
    return img_rescale


config = configparser.ConfigParser()
config.read('ae-config.txt')

Model_Name = config['ae_settings']['Model_Name']
side = int(config['ae_settings']['side'])
b = int(config['ae_settings']['b'])
cm = 'grayscale' if b==1 else 'rgb'
clusters = int(config['ae_settings']['clusters'])


valid_dir = '/data/astroml/aspindler/AstroSense/Data/Scratch/GalZoo2/Images/Test'

datagen = ImageDataGenerator(rescale=1./255,
                             preprocessing_function=contrast_enhance)
test_generator = datagen.flow_from_directory(valid_dir, target_size=(side,side),
                                              color_mode=cm, class_mode='sparse',
                                              batch_size=50)

encoder = models.load_model('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_encoder'+Model_Name+'.h5')
decoder = models.load_model('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Models/trained_decoder'+Model_Name+'.h5')

z_pred = encoder.predict_generator(test_generator, verbose=1)
B = BayesianGaussianMixture(n_components=clusters, verbose=1, covariance_type='diag')
B.fit(z_pred)
assert B.converged_ == True
c_pred = B.predict(z_pred)

for i in range(0,clusters):
    print(i)
    f, ax = plt.subplots(10,10, figsize=(15,15))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=0, hspace=0)
    for j in range(0,min(100,np.sum(c_pred==i))):
        o = decoder.predict(z_pred[c_pred==i][j][None,:])
        ax.ravel()[j].imshow(o[0,:,:,0], vmin=0, vmax=1, cmap='binary')
    [[a.axis('off') for a in ax.ravel()]]
    plt.savefig('/data/astroml/aspindler/AstroVaDEr/SCRATCH/Figures/Clusters/'+Model_Name+'_cluster_'+str(i)+'.png')
    plt.close()