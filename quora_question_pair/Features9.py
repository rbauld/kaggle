# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import feat_gen
import importlib; importlib.reload(feat_gen)
from nltk import word_tokenize
from gensim.models import KeyedVectors
from tqdm import tqdm, tqdm_pandas
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
stop_words = stopwords.words('english')

# Load and clean data #########################################################
#print('loading original data.....')
#train_orig =  pd.read_csv('./input/train.csv', header=0)
#test_orig =  pd.read_csv('./input/test.csv', header=0)

train_df =  pd.read_csv('./input/train.csv', header=0)
test_df =  pd.read_csv('./input/test.csv', header=0)

train_df.fillna(' ', inplace=True)
test_df.fillna(' ', inplace=True)

# Clean data
train_df['q1_clean'] = train_df['question1'].apply(feat_gen.clean1)
train_df['q2_clean'] = train_df['question2'].apply(feat_gen.clean1)

# Comment out
#from gensim.scripts.glove2word2vec import glove2word2vec
#glove2word2vec('./word2Vec_models/glove.840B.300d.txt', './word2Vec_models/glove_w2vec.txt')

# from gensim.models import Word2Vec

model = KeyedVectors.load_word2vec_format('./word2Vec_models/glove_w2vec.txt')

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

    
question1_vectors = np.zeros((train_df.shape[0], 300))
question2_vectors = np.zeros((train_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(train_df.q1_clean.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((train_df.shape[0], 300))
for i, q in tqdm(enumerate(train_df.q2_clean.values)):
    question2_vectors[i, :] = sent2vec(q)
     
train_df['cosine_distance2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['cityblock_distance2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['jaccard_distance2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['canberra_distance2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['euclidean_distance2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['minkowski_distance2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['braycurtis_distance2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['skew_q1vec2'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
train_df['skew_q2vec2'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
train_df['kur_q1vec2'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
train_df['kur_q2vec2'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

test_df['q1_clean'] = test_df['question1'].apply(feat_gen.clean1)
test_df['q2_clean'] = test_df['question2'].apply(feat_gen.clean1)

# Test set
# model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
question1_vectors = np.zeros((test_df.shape[0], 300))
question2_vectors = np.zeros((test_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(test_df.q1_clean.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((test_df.shape[0], 300))
for i, q in tqdm(enumerate(test_df.q2_clean.values)):
    question2_vectors[i, :] = sent2vec(q)
     
test_df['cosine_distance2'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['cityblock_distance2'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['jaccard_distance2'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['canberra_distance2'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['euclidean_distance2'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['minkowski_distance2'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['braycurtis_distance2'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['skew_q1vec2'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
test_df['skew_q2vec2'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
test_df['kur_q1vec2'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
test_df['kur_q2vec2'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]    
    
    
feat_names = ['cosine_distance2', 'cityblock_distance2', 'jaccard_distance2',
              'canberra_distance2', 'euclidean_distance2', 'minkowski_distance2',
              'braycurtis_distance2', 'skew_q1vec2', 'skew_q2vec2',
              'kur_q1vec2', 'kur_q2vec2']    
    
del question1_vectors, question2_vectors    
    
import pickle
f = open('train_F8.pickle', 'rb')
train_df2 = pickle.load(f)
f.close()

f = open('test_F8.pickle', 'rb')
test_df2 = pickle.load(f)
f.close()


train_df2 = pd.concat([train_df2, train_df[feat_names]], axis = 1)
test_df2 = pd.concat([test_df2, test_df[feat_names]], axis = 1)

# del train_df, test_df 
    
    
train_df2.to_csv('train_F9.csv')    
test_df2.to_csv('test_F9.csv')    