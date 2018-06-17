#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: vaecf_evaluate.py
Author: leowan(leowan)
Date: 2018/06/12 17:46:30
"""

import numpy as np
from scipy import sparse
import tensorflow as tf

import vaecf as model
import vaecf_metric as metric

# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2


def predict(base_sididx_arr, n_items):
    # layer node num
    p_dims = [200, 600, n_items]

    tf.reset_default_graph()
    vae = model.MultiVAE(p_dims, lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph()

    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    chkpt_dir = './chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    
    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))
        # history repr
        X = sparse.csr_matrix((np.ones_like(base_sididx_arr), (np.zeros_like(base_sididx_arr), base_sididx_arr)), shape=(1, n_items), dtype=np.int16)
        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')
        # predict multinomial
        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        return pred_val


def evaluate(n_users, n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te):
    # layer node num
    p_dims = [200, 600, n_items]

    tf.reset_default_graph()
    vae = model.MultiVAE(p_dims, lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph()

    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    chkpt_dir = './chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)
    
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)

    # validation batch size (since the entire validation set might not fit into GPU memory)
    batch_size_test = 2000

    print("chkpt directory: %s" % chkpt_dir)

    n100_list, r20_list, r50_list = [], [], []

    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            n100_list.append(metric.NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
            r20_list.append(metric.Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
            r50_list.append(metric.Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
