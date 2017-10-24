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
print('loading data.....')
train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

train_df = train_orig.copy()
test_df = test_orig.copy()

print('cleaning data.....')
train_df['question1'].fillna('', inplace=True)
train_df['question2'].fillna('', inplace=True)
test_df['question1'].fillna('', inplace=True)
test_df['question2'].fillna('', inplace=True)


train_df['q1_clean1'] = train_df.apply(lambda x: feat_gen.clean1(x['question1']), axis=1)
train_df['q2_clean1'] = train_df.apply(lambda x: feat_gen.clean1(x['question2']), axis=1)
test_df['q1_clean1'] = test_df.apply(lambda x: feat_gen.clean1(x['question1']), axis=1)
test_df['q2_clean1'] = test_df.apply(lambda x: feat_gen.clean1(x['question2']), axis=1)

# Build features ##############################################################
print('magic features.....')
(train_1, test_1) = feat_gen.magic1(train_df, test_df)
train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)
del train_1, test_1

###############################################################################
print('wordmatch features.....')
(train_1, test_1) = feat_gen.wordmatch1(train_df, test_df)
(train_2, test_2) = feat_gen.wordmatch1(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
# Drop redundant columns
test_df = test_df.drop('id',1)

###############################################################################
print('ngram features.........')
(train_1, test_1) = feat_gen.ngram_stats2(train_df, test_df)
(train_2, test_2) = feat_gen.ngram_stats2(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
print('edit_distance....')
(train_1, test_1) = feat_gen.edit_distance(train_df, test_df)
(train_2, test_2) = feat_gen.edit_distance(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

###############################################################################
print('fuzzy_feats....')
(train_1, test_1) = feat_gen.fuzzy_feats(train_df, test_df)
(train_2, test_2) = feat_gen.fuzzy_feats(train_df, test_df, qcolumns = ['q1_clean1','q2_clean1'], append='_c')

train_df = train_df.combine_first(train_1)
test_df  = test_df.combine_first(test_1)

train_df = train_df.combine_first(train_2)
test_df  = test_df.combine_first(test_2)
del train_1, test_1, train_2, test_2

print('saving data....')
f = open('train_F2.pickle', 'wb') 
pickle.dump(train_df, f)
f.close()

f = open('test_F2.pickle', 'wb') 
pickle.dump(test_df, f)
f.close()


