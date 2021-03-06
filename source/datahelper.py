import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import PredefinedSplit, KFold, ParameterGrid

## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

# CHARPROTSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
#             'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
#             'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
#             'O': 20, 'U': 20,
#             'B': (2, 11),
#             'Z': (3, 13),
#             'J': (7, 9) }
# CHARPROTLEN = 21

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 

#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1 

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind))) 
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()



## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
  def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle = False):
    self.SEQLEN = seqlen
    self.SMILEN = smilen
    #self.NCLASSES = n_classes
    self.charseqset = CHARPROTSET
    self.charseqset_size = CHARPROTLEN

    self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
    self.charsmiset_size = CHARISOSMILEN
    self.PROBLEMSET = setting_no

    # read raw file
    self._raw = self.read_sets( fpath, setting_no )

    # iteration flags
    self._num_data = len(self._raw)


  def read_sets(self, fpath, setting_no): ### fpath should be the dataset folder /kiba/ or /davis/
    print("Reading %s start" % fpath)

    test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
    train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))
    
    return test_fold, train_folds

  def parse_data(self, fpath,  with_label=True): 
		
    print("Read %s start" % fpath)

    ligands = json.load(open(fpath+"ligands_iso.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath+"proteins.txt"), object_pairs_hook=OrderedDict)
    import sys
    if sys.version_info >= (3, 0):
        Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1') ### TODO: read from raw
    else:
        Y = pickle.load(open(fpath + "Y","rb"))
    
    XD = []
    XT = []

    if with_label:
        for d in ligands.keys():
            XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
    else:
        for d in ligands.keys():
            XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))
  
    return XD, XT, Y


def get_DTC_train(data_file, max_smi_len, max_seq_len, with_label=True):
    dtc_train = pd.read_csv(data_file)
    
    dtc_train.drop('Unnamed: 0', axis=1, inplace=True)
    
    if with_label:
        dtc_train = dtc_train.groupby(['smiles', 'fasta']).aggregate(np.median).reset_index()

    for ind in dtc_train[dtc_train['smiles'].str.contains('\n')].index:
        dtc_train.loc[ind, 'smiles'] = dtc_train.loc[ind, 'smiles'].split('\n')[0]

    n_samples = dtc_train['smiles'].shape[0]
    XD, XT = [1 for i in range(n_samples)], [1 for i in range(n_samples)]

    for d in dtc_train['smiles'].unique():
        labeled_smiles = label_smiles(d, max_smi_len, CHARISOSMISET)
        indices = np.where(dtc_train['smiles']==d)[0]

        for ind in indices:
            XD[ind] = labeled_smiles

    for t in dtc_train['fasta'].unique():
        labeled_sequence = label_sequence(t, max_seq_len, CHARPROTSET)
        indices = np.where(dtc_train['fasta']==t)[0]

        for ind in indices:
            XT[ind] = labeled_sequence

    if with_label:
        return XD, XT, dtc_train['value'].values
    else:
        return XD, XT

def load_data(FLAGS):
    dataset = DataSet( fpath = FLAGS.dataset_path,
                      setting_no = FLAGS.problem_type,
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size

    XD, XT, Y = dataset.parse_data(fpath = FLAGS.dataset_path)
    XD, XT, Y = np.asarray(XD), np.asarray(XT), np.asarray(Y)

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)  #basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    Y = np.mat(np.copy(Y))
    train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, label_row_inds, label_col_inds)


    XD_dtc, XT_dtc, Y_dtc = get_DTC_train(FLAGS.dtc_data_file, FLAGS.max_smi_len, FLAGS.max_seq_len)

    all_train_drugs = np.concatenate((np.asarray(train_drugs), np.asarray(XD_dtc)), axis=0)
    all_train_prots = np.concatenate((np.asarray(train_prots), np.asarray(XT_dtc)), axis=0)
    all_train_Y = np.concatenate((np.asarray(train_Y), np.asarray(Y_dtc)), axis=0)
    all_train_Y = -np.log10(1e-8+all_train_Y/1e9)

    np.random.seed(15)
    #reduce_inactive_couples = np.concatenate((np.random.choice(np.where(all_train_Y<5)[0], 10000, False), np.where(all_train_Y>=5)[0]))
    #all_train_drugs, all_train_prots, all_train_Y = all_train_drugs[reduce_inactive_couples], all_train_prots[reduce_inactive_couples],\
    #                                all_train_Y[reduce_inactive_couples]

    return all_train_drugs, all_train_prots, all_train_Y    
    
def get_n_fold_by_drugs(all_drugs, n_splits=5):
    unique_drugs = np.unique(all_drugs, axis=0)
    test_folds = np.ones(all_drugs.shape[0])
    kf = KFold(n_splits, random_state=15)
    
    j = 0
    for _, validation_drugs in kf.split(np.arange(unique_drugs.shape[0])):
        val_inds = []

        for drug_ind in validation_drugs:
            willbe_added =  list(np.where((~(all_drugs==unique_drugs[drug_ind, :])).sum(axis=1) == 0)[0])
            val_inds +=   willbe_added
        test_folds[val_inds] = j
        j += 1
    
    return PredefinedSplit(test_folds)