# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import feat_gen
import importlib; importlib.reload(feat_gen)
import pickle


# Load and clean data #########################################################
print('loading original data.....')
train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

print('loading F2 features....')
f = open('train_F2.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F2.pickle', 'rb')
test_df = pickle.load(f)
f.close()

###############################################################################
print('ngram char features.........')
(train_1, test_1) = feat_gen.ngram_stats2(train_df, test_df, append = 'char', char=True)
(train_2, test_2) = feat_gen.ngram_stats2(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='char_c', char=True)

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
print('saving data....')

import pickle
f = open('train_F3.pickle', 'wb') 
pickle.dump(train_df, f)
f.close()

f = open('test_F3.pickle', 'wb') 
pickle.dump(test_df, f)
f.close()
