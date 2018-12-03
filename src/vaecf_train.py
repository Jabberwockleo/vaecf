#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: vaecf_train.py
Author: leowan(leowan)
Date: 2018/06/12 11:14:05
"""

import os
import shutil
import numpy as np
from scipy import sparse
import tensorflow as tf

import matplotlib.pyplot as plt
IS_ILLUSTRATED = True

import vaecf as model
import vaecf_metric as metric

def train(n_users, n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te):
    N = train_data.shape[0]
    idxlist = list(range(N))

    # training batch size
    batch_size = 500
    batches_per_epoch = int(np.ceil(float(N) / batch_size))

    N_vad = vad_data_tr.shape[0]
    idxlist_vad = list(range(N_vad))

    # validation batch size (since the entire validation set might not fit into GPU memory)
    batch_size_vad = 500

    # the total number of gradient updates for annealing
    total_anneal_steps = 200000
    # largest annealing parameter
    anneal_cap = 0.2

    # layer node num
    p_dims = [200, 600, n_items]
    
    # epoch num
    n_epochs = 200

    tf.reset_default_graph()
    vae = model.MultiVAE(p_dims, lam=0.0, random_seed=98765)

    saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    log_dir = './log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    print("log directory: %s" % log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    chkpt_dir = './chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps/1000, anneal_cap, arch_str)

    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)

    print("chkpt directory: %s" % chkpt_dir)

    ndcgs_vad = []

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf

        update_count = 0.0

        for epoch in range(n_epochs):
            print('epoch:{}'.format(epoch))
            np.random.shuffle(idxlist)
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                if bnum % 10 == 0:
                    print('  batch_num:{} start_index:{} end_index:{} N_vad:{}'.format(bnum, st_idx, end_idx, N_vad))
                X = train_data[idxlist[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap

                feed_dict = {vae.input_ph: X,
                             vae.keep_prob_ph: 0.5,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1}
                sess.run(train_op_var, feed_dict=feed_dict)

                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_train,
                                               global_step=epoch * batches_per_epoch + bnum)

                update_count += 1

            # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                if bnum % 10 == 0:
                    print('  batch_num:{} start_index:{} end_index:{} N_vad:{}'.format(bnum, st_idx, end_idx, N_vad))
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X} )
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(metric.NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)

            # update the best model (if necessary)
            print('  cur_ndcg:{} best_ndcg:{}'.format(ndcg_, best_ndcg))
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_ndcg = ndcg_

    if IS_ILLUSTRATED:
        fig = plt.figure(figsize=(12, 3))
        plt.plot(ndcgs_vad)
        plt.ylabel("Validation NDCG@100")
        plt.xlabel("Epochs")
        fig.savefig('training_curve.png')
        plt.close(fig)
    return vae
