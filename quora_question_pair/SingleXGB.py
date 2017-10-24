# -*- coding: utf-8 -*-
"""
Created on Fri May 12 19:00:17 2017

@author: Reg
"""
import pandas as pd 
import gc
import xgboost as xgb
from math import floor
from time import time

def resample(df_in, lb_target = 0.175):
    num_pos = len(df_in[df_in['is_duplicate'] == 1])
    num_neg = len(df_in[df_in['is_duplicate'] == 0])    
    alpha = (num_pos/num_neg)*(1/lb_target - 1)
    
    neg_set = df_in[df_in['is_duplicate'] == 0].sample(frac=alpha, replace=True)
    pos_set = df_in[df_in['is_duplicate'] == 1]
    
    df_in_corr = pd.concat([pos_set, neg_set])
    df_in_corr = df_in_corr.sample(frac=1)    
    return df_in_corr

# load data
print('load data....')
train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

train_df =  pd.read_csv('train_F9.csv', header=0, encoding='ISO-8859-1')
test_df =  pd.read_csv('test_F9.csv', header=0, encoding='ISO-8859-1')


# Define feature lists, for indexing
non_feat = list(train_orig.columns)+list(test_orig.columns)+list(['q2_clean1','q1_clean1'])
feat_cols = [x for x in test_df.columns if x not in non_feat]
del train_orig, test_orig

# Best features so far
feat_list = feat_cols[1:]

# Keep only some features
train_df = train_df[feat_list+['is_duplicate']]
test_df = test_df[feat_list+['test_id']]
gc.collect()

###############################################################################
# Build validation and training sets

print('build training/validation set....')
valid_df = train_df.iloc[floor(len(train_df)*0.9):]
train_df = train_df.iloc[:floor(len(train_df)*0.9)]

# Check class balance for each data set
print(valid_df['is_duplicate'].mean())
print(train_df['is_duplicate'].mean())

# Reballance both datasets
valid_df = resample(valid_df)
train_df = resample(train_df)

# Check
print(valid_df['is_duplicate'].mean())
print(train_df['is_duplicate'].mean())

# Train model
x_train = train_df[feat_list].values
y_train = train_df['is_duplicate'].values

x_valid = valid_df[feat_list].values
y_valid = valid_df['is_duplicate'].values

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['max_depth'] = 8
params['eta'] = 0.02   

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

print('train first model....')
start_time = time()
bst = xgb.train(params, d_train, 50000, watchlist,early_stopping_rounds=50, verbose_eval=10)
print(time()-start_time)

# This is used to determine the best round to stop on. This is not the ideal traning method,
# but it does allow training the model on the entire data set, which gave some gains over
# using a only a train/test method + early stopping.
n_iter = bst.best_iteration

## Fit to entire training set ################################################

print('training final model....')
train_df =  pd.read_csv('train_F9.csv', header=0, encoding='ISO-8859-1')
train_df = train_df[feat_list+['is_duplicate']]
gc.collect()

train_df = resample(train_df)
x_train = train_df[feat_list].values
y_train = train_df['is_duplicate'].values

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['max_depth'] = 8
params['eta'] = 0.02 

d_train = xgb.DMatrix(x_train, label=y_train)
watchlist = [(d_train, 'train')]

start_time = time()
bst = xgb.train(params, d_train, n_iter, watchlist,early_stopping_rounds=10, verbose_eval=10)
print(time()-start_time)

print('preparing submission....')
test_X = test_df[feat_list].values
d_test = xgb.DMatrix(test_X)
p_test = bst.predict(d_test)

# Prepare submission file
sub = pd.DataFrame()
sub['test_id'] = test_df['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('submit.csv', index=False)