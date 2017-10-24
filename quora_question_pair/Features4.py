# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import feat_gen
import importlib; importlib.reload(feat_gen)
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict

# Load and clean data #########################################################
#print('loading original data.....')
#train_orig =  pd.read_csv('./input/train.csv', header=0)
#test_orig =  pd.read_csv('./input/test.csv', header=0)

print('loading F3 features....')
f = open('train_F3.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F3.pickle', 'rb')
test_df = pickle.load(f)
f.close()

###############################################################################
print('building nieve bayes features')
maxNumFeatures = 30000

# bag of letter sequences (chars)
BagOfWordsExtractor1 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(1,1), 
                                      binary=True, lowercase=True)

BagOfWordsExtractor2 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(2,2), 
                                      binary=True, lowercase=True)

BagOfWordsExtractor3 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(3,3), 
                                      binary=True, lowercase=True)

BagOfWordsExtractor1234 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(1,4), 
                                      binary=True, lowercase=True)

BOW_ExList = [BagOfWordsExtractor1,BagOfWordsExtractor2,BagOfWordsExtractor3,BagOfWordsExtractor1234]
BOW_labels = ['1','2','3','1234']

qcols = ['question1', 'question2']

for i, extr in enumerate(BOW_ExList):
    # Build vectorizors and transform data
    print(i)
    extr.fit(pd.concat((train_df[qcols[0]],train_df[qcols[1]])).unique())
    BOW_q1_chgram = extr.transform(train_df[qcols[0]])
    BOW_q2_chgram = extr.transform(train_df[qcols[1]])
    test_BOW_q1_chgram = extr.transform(test_df[qcols[0]])
    test_BOW_q2_chgram = extr.transform(test_df[qcols[1]])   
    
    # make features
    BPW_chgram_add = BOW_q1_chgram + BOW_q2_chgram
    BPW_chgram_intersec = BOW_q1_chgram.multiply(BOW_q2_chgram).sign()
    test_BPW_chgram_add = test_BOW_q1_chgram + test_BOW_q2_chgram
    test_BPW_chgram_intersec = test_BOW_q1_chgram.multiply(test_BOW_q2_chgram).sign()
    del BOW_q1_chgram, BOW_q2_chgram, test_BOW_q1_chgram, test_BOW_q2_chgram
    
    # Predict test/training features
    model = MultinomialNB(alpha=1)
    y_data = train_df['is_duplicate'].values    
    
    # Train
    train_df['nBayes' +BOW_labels[i] +'_add'] = cross_val_predict(model, BPW_chgram_add, y_data, method='predict_proba', cv=5)[:,1]
    train_df['nBayes' +BOW_labels[i] +'_intersec'] = cross_val_predict(model, BPW_chgram_intersec, y_data, method='predict_proba', cv=5)[:,1]
    
    # Test
    model.fit(BPW_chgram_add, y_data)
    test_df['nBayes' +BOW_labels[i] +'_add'] = model.predict_proba(test_BPW_chgram_add)[:,1]
    
    model.fit(BPW_chgram_intersec, y_data)
    test_df['nBayes' +BOW_labels[i] +'_intersec'] = model.predict_proba(test_BPW_chgram_intersec)[:,1]
 
# Cleanup 
del BPW_chgram_add, BPW_chgram_intersec, test_BPW_chgram_add, test_BPW_chgram_intersec
del BagOfWordsExtractor1, BagOfWordsExtractor2, BagOfWordsExtractor3, BagOfWordsExtractor1234

# bag of words (words)

BOW_extr_word1 = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 
                                      analyzer='word', ngram_range=(1,3), 
                                      binary=True, lowercase=True)

BOW_ExList = [BOW_extr_word1]
BOW_labels = ['1']

qcols = ['q1_clean1', 'q2_clean1']

for i, extr in enumerate(BOW_ExList):
    # Build vectorizors and transform data
    print(i)
    extr.fit(pd.concat((train_df[qcols[0]],train_df[qcols[1]])).unique())
    BOW_q1_chgram = extr.transform(train_df[qcols[0]])
    BOW_q2_chgram = extr.transform(train_df[qcols[1]])
    test_BOW_q1_chgram = extr.transform(test_df[qcols[0]])
    test_BOW_q2_chgram = extr.transform(test_df[qcols[1]])   
    
    # make features
    BPW_chgram_add = BOW_q1_chgram + BOW_q2_chgram
    BPW_chgram_intersec = BOW_q1_chgram.multiply(BOW_q2_chgram).sign()
    test_BPW_chgram_add = test_BOW_q1_chgram + test_BOW_q2_chgram
    test_BPW_chgram_intersec = test_BOW_q1_chgram.multiply(test_BOW_q2_chgram).sign()
    del BOW_q1_chgram, BOW_q2_chgram, test_BOW_q1_chgram, test_BOW_q2_chgram
    
    # Predict test/training features
    model = MultinomialNB(alpha=1)
    y_data = train_df['is_duplicate'].values    
    
    # Train
    train_df['nBayes_w' +BOW_labels[i] +'_add'] = cross_val_predict(model, BPW_chgram_add, y_data, method='predict_proba', cv=5)[:,1]
    train_df['nBayes_w' +BOW_labels[i] +'_intersec'] = cross_val_predict(model, BPW_chgram_intersec, y_data, method='predict_proba', cv=5)[:,1]
    
    # Test
    model.fit(BPW_chgram_add, y_data)
    test_df['nBayes_w' +BOW_labels[i] +'_add'] = model.predict_proba(test_BPW_chgram_add)[:,1]
    
    model.fit(BPW_chgram_intersec, y_data)
    test_df['nBayes_w' +BOW_labels[i] +'_intersec'] = model.predict_proba(test_BPW_chgram_intersec)[:,1]

# Cleanup 
del BPW_chgram_add, BPW_chgram_intersec, test_BPW_chgram_add, test_BPW_chgram_intersec
del BOW_extr_word1

cols = ['nBayes1_add','nBayes2_add','nBayes3_add',
        'nBayes1_intersec', 'nBayes2_intersec','nBayes3_intersec',
        'nBayes1234_add','nBayes1234_intersec',
        'nBayes_w1_add', 'nBayes_w1_intersec', 'is_duplicate']

train_df[cols].corr()

print('saving data....')

f = open('train_F4.pickle', 'wb') 
pickle.dump(train_df, f)
f.close()

f = open('test_F4.pickle', 'wb') 
pickle.dump(test_df, f)
f.close()












