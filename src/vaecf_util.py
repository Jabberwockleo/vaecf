#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: vaecf_util.py
Author: leowan
Date: 2018/06/10 20:17:56
"""
import os
from collections import defaultdict

import numpy as np
from scipy import sparse


DATA_DIR = '../data/'


def load_input_data(fn_data, n_items, split_train_test=True):
    """
        Model input data loader
        Params:
            fn_data : data file name
            shape : (n_users, n_items)
    """
    import csv
    rows = []
    rows_reidx = []
    cols = []
    cols_tr = []
    cols_te = []
    uididx_sididx_dict = defaultdict(list)
    with open(fn_data, 'r') as fd:
        elems = csv.reader(fd, delimiter='\t')
        for uididx, sididx in elems:
            rows.append(int(uididx))
            cols.append(int(sididx))
            uididx_sididx_dict[int(uididx)].append(int(sididx))
    uniq_uididx = set(rows)
    n_users = len(uniq_uididx)
    uididx_sort = sorted(list(uniq_uididx))
    uid_reidx_dict = dict((uididx, i) for i, uididx in enumerate(uididx_sort))
    if not split_train_test:
        rows_reidx_tr = [uid_reidx_dict[uididx] for uididx in rows]
        cols_tr = cols
    else:
        rows_reidx_tr = []
        rows_reidx_te = []
        for uididx, sididx_arr in uididx_sididx_dict.items():
            sididx_arr = list(np.random.permutation(sididx_arr))
            uid_reidx = uid_reidx_dict[uididx]
            cnt_tr = int(len(sididx_arr) * 0.8)
            for sididx in sididx_arr[:cnt_tr]:
                rows_reidx_tr.append(uid_reidx)
                cols_tr.append(sididx)
            for sididx in sididx_arr[cnt_tr:]:
                rows_reidx_te.append(uid_reidx)
                cols_te.append(sididx)
    
    data_tr = sparse.csr_matrix((np.ones_like(rows_reidx_tr), (rows_reidx_tr, cols_tr)), dtype=np.int16, shape=(n_users, n_items))
    data_te = None
    if split_train_test:
        data_te = sparse.csr_matrix((np.ones_like(rows_reidx_te), (rows_reidx_te, cols_te)), dtype=np.int16, shape=(n_users, n_items))
    return data_tr, data_te


def yield_uid_sids_from_file(fn):
    """
        Generator
    """
    fd = open(fn, "r")
    for line in fd:
        elems = line.rstrip().split(',')
        uid = elems[0]
        sids = set()
        for sid in elems[1].split('|'):
            if sid != '':
                sids.add(sid)
        yield uid, sids
    fd.close()


def make_sid_count_dict(fn):
    """
        Count
    """
    from collections import defaultdict
    sid_cnt_dict = defaultdict(int)
    for uid, sids in yield_uid_sids_from_file(fn):
        for sid in sids:
            sid_cnt_dict[sid] += 1
    return sid_cnt_dict


def make_sid_index(sid_cnt_dict):
    """
        Sort
    """
    import operator
    return sorted(sid_cnt_dict.items(), key=operator.itemgetter(1), reverse=True)


def make_index(fn, min_cnt=5, max_sid_cnt=50000):
    """
        Index
    """
    print('filter out items watched by and users watches less than:{}'.format(min_cnt))
    idx2uid = {}
    uid2idx = {}
    idx2sid = {}
    sid2idx = {}
    sid_index_arr = make_sid_index(make_sid_count_dict(fn))
    idx = 0
    sid_index_arr = sorted(sid_index_arr, key=lambda x:[1], reverse=True)[:max_sid_cnt]
    for sid, cnt in sid_index_arr:
        if cnt < min_cnt:
            continue
        idx2sid[idx] = sid
        sid2idx[sid] = idx
        idx += 1
    idx = 0
    for uid, sids in yield_uid_sids_from_file(fn):
        cnt = 0
        for sid in sids:
            if sid in sid2idx:
                cnt += 1
        if cnt < min_cnt:
            continue
        idx2uid[idx] = uid
        uid2idx[uid] = idx
        idx += 1
    return tuple([idx2uid, uid2idx, idx2sid, sid2idx])


def make_uid_sids_dict(fn):
    """
        uid -> sids
    """
    uid_sids_dict = {}
    for uid, sids in yield_uid_sids_from_file(fn):
        uid_sids_dict[uid] = sids
    return uid_sids_dict


def make_train_dev_test_files(idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict, data_dir=DATA_DIR):
    """
        Create train dev files from uid sids info
    """
    def write_file(uid_arr, fd):
        '''
            Write train/dev/test files
        '''
        for uid in uid_arr:
            raw_sids = uid_sids_dict[uid]
            if uid not in uid2idx:
                continue
            sids = []
            for sid in raw_sids:
                if sid in sid2idx:
                    sids.append(sid)
            pos_sids = list(sids)
            pos_ratings = np.ones(len(pos_sids), dtype=int)
            X = pos_sids
            y = pos_ratings.tolist()
            for xt, yt in zip(X, y):
                fd.write('{}\t{}\n'.format(uid2idx[uid], sid2idx[xt]))

    from sklearn.model_selection import train_test_split
    import numpy as np
    fn_train = os.path.join(data_dir, 'data_train.txt')
    fn_dev = os.path.join(data_dir, 'data_dev.txt')
    fn_test = os.path.join(data_dir, 'data_test.txt')
    fdt = open(fn_train, 'w')
    fdv = open(fn_dev, 'w')
    fdd = open(fn_test, 'w')
    # randomize
    uids = list(np.random.permutation(list(uid2idx.keys())))
    # count split
    dev_cnt = int(len(uids) * 0.1)
    test_cnt = dev_cnt
    train_cnt = len(uids) - dev_cnt - test_cnt
    # split uids
    train_uids = uids[:train_cnt]
    dev_uids = uids[train_cnt:train_cnt + dev_cnt]
    test_uids = uids[-test_cnt:]
    # write files
    write_file(train_uids, fdt)
    write_file(dev_uids, fdv)
    write_file(test_uids, fdd)
    fdt.close()
    fdv.close()
    fdd.close()
    return fn_train, fn_dev, fn_test


def count_user_item_entry(fn):
    """
        Count user item entry in file
    """
    import csv
    entry_cnt = 0
    max_user_idx = 0
    max_item_idx = 0
    with open(fn, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uididx, sididx in elems:
            entry_cnt += 1
            max_user_idx = max(max_user_idx, int(uididx))
            max_item_idx = max(max_item_idx, int(sididx))
    return max_user_idx, max_item_idx, entry_cnt
