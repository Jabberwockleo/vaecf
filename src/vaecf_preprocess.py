#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: vaecf_preprocess.py
Author: leowan(leowan)
Date: 2018/06/12 10:59:34
"""

import os
import vaecf_util as util

DATA_DIR = '../data/'
fn_uid_sids = 'uid_sids.txt'

def preprocess_data(data_dir=DATA_DIR, fn_uid_sids=fn_uid_sids):
    idx2uid, uid2idx, idx2sid, sid2idx = util.make_index(os.path.join(DATA_DIR, fn_uid_sids), max_sid_cnt=10000)
    uid_sids_dict = util.make_uid_sids_dict(os.path.join(DATA_DIR, fn_uid_sids))
    fn_train, fn_dev, fn_test = util.make_train_dev_test_files(
        idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict)
    print(fn_train, fn_dev, fn_test)
    n_users = len(uid2idx)
    print('n_users:{}'.format(n_users))
    n_items = len(sid2idx)
    print('n_items:{}'.format(n_items))

    train_data, _ = util.load_input_data(fn_train, n_items=n_items, split_train_test=False)
    vad_data_tr, vad_data_te = util.load_input_data(fn_dev, n_items=n_items, split_train_test=True)
    test_data_tr, test_data_te = util.load_input_data(fn_test, n_items=n_items, split_train_test=True)

    return tuple([idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict,
                  n_users, n_items,
                  train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te])
