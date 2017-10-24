# -*- coding: utf-8 -*-
"""
Created on Sun May 28 12:15:54 2017

@author: Reg
"""

import pandas as pd
import pickle

train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

train_orig.fillna(' ', inplace=True)
test_orig.fillna(' ', inplace=True)

f = open('train_F7_raw.pickle', 'rb')
train_raw = pickle.load(f)
f.close()

f = open('test_F7_raw.pickle', 'rb')
test_raw = pickle.load(f)
f.close()

col_names = ['evcent1', 'evcent2','evcent12' ,'evcent_diff',
             'edp','pgrnk',
             'trns_avglocal', 'trns1','trns2','trns12','trns_diff',
             'ass_deg','avg_pathlength',
             'between1','between2','between12','between_diff',
             'bibcoup','density',
             'ecc1', 'ecc2', 'ecc_avg', 'ecc_diff',
             'girth', 'maxflow', 'simm_log', 'simm_jacc']

train_raw = pd.DataFrame(train_raw, columns = col_names)
train_raw.fillna(train_raw.mean(), inplace=True) 
train_df = train_orig.combine_first(train_raw)
train_df.corr()['is_duplicate']

my_index = test_orig[test_orig['question1']!=test_orig['question2']].index
test_raw = pd.DataFrame(test_raw, columns = col_names, index=my_index)
test_raw.fillna(train_raw.mean(), inplace=True)

test_df = test_orig.copy()
test_df[col_names] = test_raw[col_names]
test_df.loc[~test_df.index.isin(my_index), col_names] = test_df.loc[~test_df.index.isin(my_index), col_names].fillna(train_df[col_names].mean())
del train_orig, test_orig, test_raw, train_raw

f = open('train_F6.pickle', 'rb')
train_df2 = pickle.load(f)
f.close()

f = open('test_F6.pickle', 'rb')
test_df2 = pickle.load(f)
f.close()

train_df2 = train_df2.combine_first(train_df)
test_df2 = test_df2.combine_first(test_df)

del train_df, test_df

 
f = open('train_F7.pickle', 'wb') 
pickle.dump(train_df2, f)
f.close()

f = open('test_F7.pickle', 'wb') 
pickle.dump(test_df2, f , protocol=4)
f.close()