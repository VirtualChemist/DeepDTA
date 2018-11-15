import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from os import path
import os,binascii
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm

from run_experiments import *
from datahelper import *
import keras.metrics
keras.metrics.cindex_score = cindex_score
from sklearn.metrics import mean_squared_error, f1_score
from keras.models import load_model
from sklearn.model_selection import ParameterGrid

FLAGS = argparser()
FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
FLAGS.dataset_path= '../data/davis/'
FLAGS.dtc_data_file = '../data/dtc_for_deepDTA.csv'
FLAGS.charseqset_size = CHARPROTLEN 
FLAGS.charsmiset_size = CHARISOSMILEN 
dtc_data_file = '../../data/dtc_for_deepDTA.csv'

FLAGS_dict = vars(FLAGS)

for i, val in FLAGS_dict.items():
    if type(FLAGS_dict[i]) != list:
        FLAGS_dict[i] = [val]
#TRAIN#
n_repeats = 2
param_grid = ParameterGrid([
    {
        'num_filters': FLAGS.num_windows,
        'smi_filter_length': FLAGS.smi_window_lengths,
        'seq_filter_length': FLAGS.seq_window_lengths,
        'dropout': FLAGS.dropouts,
        'apply_bn': FLAGS.bns
    }
])

if path.exists(FLAGS.results_pickle):
    with open(FLAGS.results_pickle, 'rb') as handle:
        results = pickle.load(handle)
else:
    results = {}

param_grid = ParameterGrid(FLAGS_dict)

if path.exists(FLAGS.results_pickle[0]):
    with open(FLAGS.results_pickle[0], 'rb') as handle:
        results = pickle.load(handle)
else:
    results = {}

for ind in range(len(param_grid)):
       
    params = param_grid[ind]
    cur_FLAGS = Namespace(**params)
    all_train_drugs, all_train_prots, all_train_Y = load_data(cur_FLAGS)
    val_inds = get_n_fold_by_drugs(all_train_drugs, 10)
    param_name = str(binascii.b2a_hex(os.urandom(15))).replace("'", '')

    early_stopping_callback = CustomStopper(monitor='val_loss', patience=30, start_epoch=50)
    fold_id, avg_loss, avg_cindex, avg_f1, avg_rmse = 0, 0.0, 0.0, 0.0, 0.0
    for tr_fold, val_fold in val_inds.split():
        if fold_id >= n_repeats:
            break

        model_name='checkpoints/davis_dtc_dta_'+param_name+'fold'+str(fold_id)

        checkpoint_callback = ModelCheckpoint(model_name+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        XD_train, XT_train, Y_train = all_train_drugs[tr_fold], all_train_prots[tr_fold], all_train_Y[tr_fold]
        XD_val, XT_val, Y_val = all_train_drugs[val_fold], all_train_prots[val_fold], all_train_Y[val_fold]

        gridmodel = build_combined_categorical(cur_FLAGS, cur_FLAGS.num_windows, cur_FLAGS.smi_window_lengths, 
                                       cur_FLAGS.seq_window_lengths, cur_FLAGS.dropouts, apply_bn=cur_FLAGS.bns)

        gridres = gridmodel.fit(([XD_train, XT_train ]), Y_train, batch_size=cur_FLAGS.batch_size, epochs=cur_FLAGS.num_epoch, 
                validation_data=( ([np.array(XD_val), np.array(XT_val) ]), np.array(Y_val))
                   ,callbacks=[early_stopping_callback, checkpoint_callback], verbose=2)
        
        gridmodel = load_model(model_name+'.h5')
        predicted_labels = gridmodel.predict([np.array(XD_val), np.array(XT_val)])[:, 0]
        loss, rperf2 = gridmodel.evaluate(([np.array(XD_val),np.array(XT_val) ]), np.array(Y_val), verbose=0)
        avg_loss += loss/n_repeats
        avg_cindex += rperf2/n_repeats
        avg_f1 += f1_score(Y_val>7, predicted_labels>7)/n_repeats
        avg_rmse += np.sqrt(mean_squared_error(Y_val, predicted_labels))/n_repeats
        fold_id += 1
    
    results[param_name] = {'FLAGS':cur_FLAGS, 'params': params, 'loss': avg_loss, 'cindex': avg_cindex,
                       'rmse':  avg_rmse, 'f1': avg_f1, 'n_repeats': n_repeats}
    with open(FLAGS.results_pickle[0], 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)