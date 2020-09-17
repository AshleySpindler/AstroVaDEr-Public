#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:27:48 2020

@author: ashleyspindler

Loss functions for AstroVader
"""

import keras
import keras.backend as K


#%%============================================================================
# Loss Functions and Metrics
def gamma_training(outputs, params):
    def gamma_loss(y_true, y_pred):

        if params['loss_type']=='binary_crossentropy':
            recon_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*params['original_dims']
        else:
            recon_loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*params['original_dims']

        kl_loss = -0.5 * K.sum(1 + outputs['z_log_var'] - K.square(outputs['z_mean']) - K.exp(outputs['z_log_var']), axis=-1)

        tot_loss = K.mean(recon_loss) + params['warm_up']*K.mean(kl_loss)

        return tot_loss
    return gamma_loss

def vae_training(outputs, params):
    def vae_loss(y_true, y_pred):
        
        if params['loss_type']=='binary_crossentropy':
            recon_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*params['original_dims']
        else:
            recon_loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*params['original_dims']
            
        kl_loss = -0.5 * K.sum(1 + outputs['z_log_var'] - K.square(outputs['z_mean']) - K.exp(outputs['z_log_var']), axis=-1)
        
        tot_loss = K.mean(recon_loss) + (K.pow(params['annealing_factor'],3)\
                                     + params['warm_up'])*K.mean(kl_loss)
        
        return tot_loss
    return vae_loss


def static_training(outputs, weights, params):
    """
    training loss for annealing and static phase training, modified from
    https://github.com/king/s3vdc to support keras (tf 1.14)
    """
    def static_loss(y_true, y_pred):
        
        log2pi = 1.8378770664093453

        if params['loss_type']=='binary_crossentropy':
            recon_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*params['original_dims']
        else:
            recon_loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*params['original_dims']
        
        
        _mu = K.tile(
            K.expand_dims(weights["mu"], 0), [params["batch_size"], 1, 1]
        )
        _sigma = K.tile(
            K.expand_dims(weights["lambda"], 0), [params["batch_size"], 1, 1]
        )
        _z_mu = K.tile(
            K.expand_dims(outputs["z_mean"], -1), [1, 1, params['clusters']]
        )
        _z_sigma = K.tile(
            K.expand_dims(outputs["z_log_var"], -1), [1, 1, params['clusters']]
        )

        ###

        latent_loss = -0.5 * K.sum(1.0 + outputs["z_log_var"], axis=1)
        latent_loss -= K.sum(
            outputs["gamma"]
            * K.log(
                K.tile(
                    K.expand_dims(weights["theta"], 0),
                    [params["batch_size"], 1],
                )
            ),
            axis=1,
        )
        latent_loss += K.sum(
            outputs["gamma"] * K.log(outputs["gamma"]), axis=1
        )
        """ Change here from Cao paper, taking latents*log2pi outside
        the sum over latents
        """
        latent_loss += 0.5 * K.sum(
            outputs['gamma']
              * (
                 params['latents'] * log2pi
                 + K.sum(
                     K.log(_sigma)
                     + (K.square(_z_mu - _mu) + K.exp(_z_sigma)) / _sigma,
                     axis=[1],
                   )
                 ),
             axis=-1,
           )
        latent_loss = K.mean(latent_loss) * params['scale_jank']

        tot_loss = K.mean(recon_loss) + latent_loss

        return tot_loss
    return static_loss

def beta_training(outputs, weights, params):
    def beta_loss(y_true, y_pred):
        
        log2pi = 1.8378770664093453        
        
        if params['loss_type']=='binary_crossentropy':
            recon_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*params['original_dims']
        else:
            recon_loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*params['original_dims']
        
        _mu = K.tile(
            K.expand_dims(weights["mu"], 0), [params["batch_size"], 1, 1]
        )
        _sigma = K.tile(
            K.expand_dims(weights["lambda"], 0), [params["batch_size"], 1, 1]
        )
        _z_mu = K.tile(
            K.expand_dims(outputs["z_mean"], -1), [1, 1, params['clusters']]
        )
        _z_sigma = K.tile(
            K.expand_dims(outputs["z_log_var"], -1), [1, 1, params['clusters']]
        )

        ###

        latent_loss = -0.5 * K.sum(1.0 + outputs["z_log_var"], axis=1)
        latent_loss -= K.sum(
            outputs["gamma"]
            * K.log(
                K.tile(
                    K.expand_dims(weights["theta"], 0),
                    [params["batch_size"], 1],
                )
            ),
            axis=1,
        )
        latent_loss += K.sum(
            outputs["gamma"] * K.log(outputs["gamma"]), axis=1
        )
        """ Change here from Cao paper, taking latents*log2pi outside
        the sum over latents
        """
        latent_loss += 0.5 * K.sum(
            outputs['gamma']
              * (
                 params['latents'] * log2pi
                 + K.sum(
                     K.log(_sigma)
                     + (K.square(_z_mu - _mu) + K.exp(_z_sigma)) / _sigma,
                     axis=[1],
                   )
                 ),
             axis=-1,
           )
        latent_loss = K.mean(latent_loss) * params['scale_jank']
        
        latent_loss = K.mean(recon_loss) + (K.pow(params['annealing_factor'],3)\
                                     + params['warm_up'])*latent_loss

        return latent_loss
    return beta_loss

def beta_training2(outputs, weights, params):
    def beta_loss(y_true, y_pred):
        
        log2pi = 1.8378770664093453        
        
        _mu = K.tile(
            K.expand_dims(weights["mu"], 0), [params["batch_size"], 1, 1]
        )
        _sigma = K.tile(
            K.expand_dims(weights["lambda"], 0), [params["batch_size"], 1, 1]
        )
        _z_mu = K.tile(
            K.expand_dims(outputs["z_mean"], -1), [1, 1, params['clusters']]
        )
        _z_sigma = K.tile(
            K.expand_dims(outputs["z_log_var"], -1), [1, 1, params['clusters']]
        )

        ###

        latent_loss = -0.5 * K.sum(1.0 + outputs["z_log_var"], axis=1)
        latent_loss -= K.sum(
            outputs["gamma"]
            * K.log(
                K.tile(
                    K.expand_dims(weights["theta"], 0),
                    [params["batch_size"], 1],
                )
            ),
            axis=1,
        )
        latent_loss += K.sum(
            outputs["gamma"] * K.log(outputs["gamma"]), axis=1
        )
        """ Change here from Cao paper, taking latents*log2pi outside
        the sum over latents
        """
        latent_loss += 0.5 * K.sum(
            outputs['gamma']
              * (
                 params['latents'] * log2pi
                 + K.sum(
                     K.log(_sigma)
                     + (K.square(_z_mu - _mu) + K.exp(_z_sigma)) / _sigma,
                     axis=[1],
                   )
                 ),
             axis=-1,
           )
        latent_loss = K.mean(latent_loss) * params['scale_jank']
        
        latent_loss = (K.pow(params['annealing_factor'],3)\
                                     + params['warm_up'])*latent_loss

        return latent_loss
    return beta_loss

def KL_metric(outputs):
    def KL_loss(y_true, y_pred):
        
        kl_loss = -0.5 * K.sum(1 + outputs['z_log_var'] - K.square(outputs['z_mean']) - K.exp(outputs['z_log_var']), axis=-1)
        
        kl_loss = K.mean(kl_loss)
        
        return kl_loss
    return KL_loss

def recon_metric(params):
    def recon_loss(y_true, y_pred):
        
        if params['loss_type']=='binary_crossentropy':
            loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=(1,2,3))*params['original_dims']
        else:
            loss = K.mean(K.square(y_true-y_pred), axis=(1,2,3))*params['original_dims']
            
        return K.mean(loss)
    return recon_loss

def VaDE_metric(outputs, weights, params):
    def vade_loss(y_true, y_pred):
        
        log2pi = 1.8378770664093453        
        
        _mu = K.tile(
            K.expand_dims(weights["mu"], 0), [params["batch_size"], 1, 1]
        )
        _sigma = K.tile(
            K.expand_dims(weights["lambda"], 0), [params["batch_size"], 1, 1]
        )
        _z_mu = K.tile(
            K.expand_dims(outputs["z_mean"], -1), [1, 1, params['clusters']]
        )
        _z_sigma = K.tile(
            K.expand_dims(outputs["z_log_var"], -1), [1, 1, params['clusters']]
        )

        ###

        latent_loss = -0.5 * K.sum(1.0 + outputs["z_log_var"], axis=1)
        latent_loss -= K.sum(
            outputs["gamma"]
            * K.log(
                K.tile(
                    K.expand_dims(weights["theta"], 0),
                    [params["batch_size"], 1],
                )
            ),
            axis=1,
        )
        latent_loss += K.sum(
            outputs["gamma"] * K.log(outputs["gamma"]), axis=1
        )
        """ Change here from Cao paper, taking latents*log2pi outside
        the sum over latents
        """
        latent_loss += 0.5 * K.sum(
            outputs['gamma']
              * (
                 params['latents'] * log2pi
                 + K.sum(
                     K.log(_sigma)
                     + (K.square(_z_mu - _mu) + K.exp(_z_sigma)) / _sigma,
                     axis=[1],
                   )
                 ),
             axis=-1,
           )
        latent_loss = K.mean(latent_loss)

        return latent_loss
    return vade_loss

def log_prob(z_mu, gmm_weights, params):
    def log_prob_metric(y_true, y_pred):

        log2pi = 1.8378770664093453
        
        precisions = 1.0 / gmm_weights['lambda']
        precisions_chol = 1.0 / K.sqrt(gmm_weights['lambda'])
        _log_prob = K.sum((K.square(gmm_weights['mu']) * precisions), axis=0)
        _log_prob -= 2.0 * K.dot(z_mu, gmm_weights['mu'] * precisions)
        _log_prob += K.dot(K.square(z_mu), precisions)
        log_det_chol = K.sum(K.log(precisions_chol), axis=0)
        log_prob = -0.5 * (params['latents'] * log2pi + _log_prob) + log_det_chol
        
        log_weights = K.log(gmm_weights['theta'])
        
        score = K.logsumexp(log_prob+log_weights, axis=1)
        
        return K.mean(score)
    return log_prob_metric