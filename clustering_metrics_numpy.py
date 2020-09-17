#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:22:24 2020

@author: ashleyspindler

Functions for unsupervised clustering analysis
"""
import numpy as np

def cluster_acc(Y_pred, Y):
    """
    Linear Assignment of cluster labels: used to compare with
    ground truth labels or between two clustering results
    Source: https://github.com/slim1017/VaDE/blob/master/VaDE.py
    """
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def metric_calinski_harabaz_numpy(X, L, num_clusters):
    """
    Copyright (C) king.com Ltd 2019
    https://github.com/king/s3vdc
    License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
    Modified by: Ashley Spindler
    %====================================================================
    The resulting Calinski-Harabaz score of sample clusters.
    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion (i.e. Covariance).
    http://www.tandfonline.com/doi/abs/10.1080/03610927408827101
    Formula of CH score for the k-th cluster:
    CH(k) = \frac{trB(k)}{trW(k)} \times \frac{k-1}{n-k},
    where n is the total number of clusters; trB(k) is the trace of the extra-cluster
    covariance matrix (i.e. the sum of feature variance); and trW(k) denotes the trace
    of intra-cluster covariance matrix.
    Largely, the higher the score is, the better the cluster result is.
    Arguments:
        X {np.array} -- 2D array of features
        L {np.array} -- 1D array of predicted labels for samples
        num_clusters {int} -- the total number of clusters
    Returns:
        np.array -- A float metric containing the Calinski-Harabaz score
    """
    num_samples = np.shape(X)[0]
    extra_dispersion, intra_dispersion = 0.0, 0.0
    mean = np.mean(X, axis=0)
    unique_L = np.unique(L)
    n_clusters_in_batch_k = np.shape(unique_L)[0]
    for k in range(num_clusters):
        _mask = np.equal(L, k)
        X_k = X[_mask]
        num_sample_k = np.shape(X_k)[0]
        if np.equal(num_sample_k,0):
            mean_k = np.zeros(np.shape(mean), dtype='float32')
        else:
            mean_k = np.mean(X_k, axis=0)
        extra_dispersion += num_sample_k * np.sum(
            np.square((mean_k - mean))
        )
        intra_dispersion += np.sum(np.square(X_k - mean_k))
    extra_disp_mean = np.mean(extra_dispersion)
    intra_disp_mean = np.mean(intra_dispersion)
    nominator = extra_disp_mean * (num_samples - n_clusters_in_batch_k)  
    denominator = intra_disp_mean * (n_clusters_in_batch_k - 1)
    if denominator == 0.0:
        _result = 1.0
    else:
        _result = nominator / denominator
    return np.mean(_result)

def metric_simplified_silhouette(C, E, p, dist_type = "euclidean"):
    """
    Copyright (C) king.com Ltd 2019
    https://github.com/king/s3vdc
    License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
    Modified by: Ashley Spindler
    %====================================================================
    Performs calculation of Silhouette Coefficient as a evaluation metric.
    Assume we obtained k clusters via certain unsupervised clustering algorithms.
    For each sample in evaluation/testing dataset, we can obtain the cluster it belongs (
    and the embedded space for some models). For the i-th sample, we calculate:
    - a(i) = mean(all distances from vector i to all other samples in the same cluster as i)
    - b(i) = min(mean distance to all samples in other clusters)
    Silhouette Coefficient = (b_i-a_i)/(max(a_i,b_i))
    To reduce the computation complexity, we adopt the following simplifications:
    - use the distance from a sample to cluster center to replace the mean distance to all
    points in a cluster.
    Arguments:
        C {np.array} -- The matrix (clusters * codes) denoting the centers of clusters.
        E {np.array} -- The embedding matrix (N * codes) for the samples.
        p {np.array} -- The column vector (of size N) containing the predicted cluster numbers.
    Keyword Arguments:
        dist_type {str} -- The type of distance metric. (default: {"euclidean"})
    Returns:
        np.array -- A float metric containing the simplified Silhouette score.
    """
    num_centroids = np.shape(C)[0]
    # calc distance matrix
    Ce = np.expand_dims(C, 1)
    Ee = np.expand_dims(E, 0)
    D = np.sqrt(np.sum(np.square(Ce-Ee), -1))
    # construct binary matrix R
    _idx = np.arange(0, num_centroids)
    R = np.equal(
            np.reshape(p, (-1, np.shape(p)[-1], 1)),
            np.reshape(_idx, (-1, 1, np.shape(_idx)[-1])),
        )
    R = np.transpose(np.squeeze(R))
    _R = np.abs(R - 1.0)
    # approximate a's and b's
    a_s = np.sum(D * R, 0)
    DmR = D * _R
    b_s = np.min(
        DmR[np.greater(DmR, 0)], 0)  # remove distance to itself
    # calc Silhouette Coefficient
    # Formula: S_i=\frac{b_i-a_i}{max\{a_i,b_i\}}
    _result = (b_s - a_s) / np.maximum(a_s, b_s)
    return np.mean(_result)

def metric_cluster_separation(A, num_clusters = None):
    """
    Copyright (C) king.com Ltd 2019
    https://github.com/king/s3vdc
    License: MIT, https://raw.github.com/king/s3vdc/LICENSE.md
    Modified by: Ashley Spindler
    %====================================================================
    Calculate the average distance between wither 2 cluster centers.
    Formula: (2/(k^2-k))\sum_{i=1}^k\sum_{j=i+1}^k||a_i-a_j||_2,
    where a_i represents the i-th row vector
    Implementation details:
    A key step of this calculation is to obtain the pairwise square distances of A,
    denoted as D. To turn it into an matrix op., we have:
    D[i,j] = (a[i]-a[j])(a[i]-a[j])'
    => D[i,j] = r[i] - 2 a[i]a[j]' + r[j],
    where r[i] is the squared norm of the i-th row of the original matrix;
    because of broadcasting, we can treat r as a column vector and hence D is
    D = r - 2 A A' + r'
    Arguments:
        A {np.array} -- The tensor of row Vectors of cluster centers
    Keyword Arguments:
        num_clusters {int} -- The number of clusters (default: {None})
    Returns:
        np.array -- A float metric containing the Cluster Separation score
    """
    r = np.sum(A * A, 1)
    r = np.reshape(r, [-1, 1])
    D = r - 2 * np.dot(A, np.transpose(A)) + np.transpose(r)
    if num_clusters is None:
        num_clusters = float(np.shape(A)[0])
    D_relu = D.copy()
    D_relu[D_relu<0] = 0
    _result = np.sum(np.sqrt(D_relu)) / (num_clusters ** 2 - num_clusters)
    return np.mean(_result)
