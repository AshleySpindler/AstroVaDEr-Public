#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:36:34 2020

@author: ashleyspindler
Callback functions for AstroVaDEr
"""

import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback

class lr_schedule:
    """
    args:
        min_lr - lr floor
        decay_steps - how often to update in epochs
    """
    def __init__(self, init_lr=1e-3, min_lr=1e-6, decay_steps=10,
                 decay_rate=0.9, staircase=False):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, epoch, lr):
        
        p = epoch/self.decay_steps
        if self.staircase:
            p = np.floor(p)

        lr_new = self.init_lr * np.power(self.decay_rate,p)

        return np.maximum(lr_new, self.min_lr)
    
class AnnealingCallback(Callback):
    def __init__(self, params):
        self.M = 1
        self.TB = params['annealing_steps']
        self.Ts = params['static_steps']
        self.Ty = params['warm_up_steps']
        self.Tm = self.Ty + (self.M-1)*(self.TB+self.Ts)
        self.annealing = params['annealing_factor']
        
    def on_epoch_begin(self, epoch, logs={}):
        
        if epoch <= self.Tm+self.TB:
            new_weight = (epoch - self.Tm) / self.TB
            K.set_value(self.annealing, new_weight)
            
            print ("KL Weight updated to " + str(K.get_value(self.annealing)**3)\
                   + " at epoch " + str(epoch))
        else:
            new_weight = 1
            K.set_value(self.annealing, new_weight)
            
        if epoch == self.Tm+self.TB+self.Ts:
            print("End of Beta Annealing Period {0}".format(self.M))
            self.M += 1
            self.Tm = self.Ty + (self.M-1)*(self.TB+self.Ts)

class MyModelCheckPoint(ModelCheckpoint):

    def __init__(self, singlemodel, *args, **kwargs):
        self.singlemodel = singlemodel
        super(MyModelCheckPoint, self).__init__(*args, **kwargs)
    def on_epoch_end(self, epoch, logs=None):
        self.model = self.singlemodel
        super(MyModelCheckPoint, self).on_epoch_end(epoch, logs)
