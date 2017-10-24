# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import time
from dask.dataframe import from_pandas
import dask
from tqdm import tqdm, tqdm_pandas
import pickle

stop_words = stopwords.words('english')


# Load and clean data #########################################################
#print('loading original data.....')
train_df =  pd.read_csv('./input/train.csv', header=0)
test_df =  pd.read_csv('./input/test.csv', header=0)

"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur
"""

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


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

###############################################################################
# Train
print('length features....')
train_df['len_char_q1'] = train_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_char_q2'] = train_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_word_q1'] = train_df.question1.apply(lambda x: len(str(x).split()))
train_df['len_word_q2'] = train_df.question2.apply(lambda x: len(str(x).split()))

#Test
test_df['len_char_q1'] = test_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_char_q2'] = test_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_word_q1'] = test_df.question1.apply(lambda x: len(str(x).split()))
test_df['len_word_q2'] = test_df.question2.apply(lambda x: len(str(x).split()))

###############################################################################
# Try paralell computation with dask
#Train
print('extra fuzzy features, train....')
train_dd = from_pandas(train_df[['question1','question2']], npartitions=8)

start_time= time.time()
train_df['fuzz_qratio']  = train_dd.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
train_df['fuzz_WRatio'] = train_dd.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
train_df['fuzz_token_set_ratio']  = train_dd.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
train_df['fuzz_token_sort_ratio'] = train_dd.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
print((time.time()-start_time))
del train_dd

#Test
print('extra fuzzy features, test....')
test_dd = from_pandas(test_df[['question1','question2']], npartitions=8)

start_time= time.time()
test_df['fuzz_qratio']  = test_dd.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
test_df['fuzz_WRatio'] = test_dd.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
test_df['fuzz_token_set_ratio']  = test_dd.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
test_df['fuzz_token_sort_ratio'] = test_dd.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=('a', np.dtype('int64'))).compute(get=dask.multiprocessing.get)
print((time.time()-start_time))
del test_dd
###############################################################################

model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)

tqdm_pandas(tqdm(desc="Train wmd:", total = len(train_df)))
train_df['wmd'] = train_df.progress_apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
tqdm_pandas(tqdm(desc="Test wmd:", total = len(test_df)))
test_df['wmd'] = test_df.progress_apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
del model

norm_model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
tqdm_pandas(tqdm(desc="Train norm wmd:", total = len(train_df)))
train_df['norm_wmd'] = train_df.progress_apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
tqdm_pandas(tqdm(desc="Test norm wmd:", total = len(test_df)))
test_df['norm_wmd'] = test_df.progress_apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
del norm_model

###############################################################################
# word2Vec features
# Train set
model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
question1_vectors = np.zeros((train_df.shape[0], 300))
question2_vectors = np.zeros((train_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(train_df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((train_df.shape[0], 300))
for i, q in tqdm(enumerate(train_df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)
     
train_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

train_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
train_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
train_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
train_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


# Test set
# model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
question1_vectors = np.zeros((test_df.shape[0], 300))
question2_vectors = np.zeros((test_df.shape[0], 300))

error_count = 0

for i, q in tqdm(enumerate(test_df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((test_df.shape[0], 300))
for i, q in tqdm(enumerate(test_df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)
     
test_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

test_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
test_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
test_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
test_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


print('saving data....')

f = open('train_F5_tmp.pickle', 'wb') 
pickle.dump(train_df, f)
f.close()

f = open('test_F5_tmp.pickle', 'wb') 
pickle.dump(test_df, f)
f.close()

# Clean features
# Train
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.loc[:,'wmd'].fillna(train_df['wmd'].mean(), inplace=True)
train_df.loc[:,'norm_wmd'].fillna(train_df['norm_wmd'].mean(), inplace=True)
train_df.loc[:,'cosine_distance'].fillna(train_df['cosine_distance'].mean(), inplace=True)
train_df.loc[:,'jaccard_distance'].fillna(train_df['jaccard_distance'].mean(), inplace=True)
train_df.loc[:,'braycurtis_distance'].fillna(train_df['braycurtis_distance'].mean(), inplace=True)

# Test
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.loc[:,'wmd'].fillna(test_df['wmd'].mean(), inplace=True)
test_df.loc[:,'norm_wmd'].fillna(test_df['norm_wmd'].mean(), inplace=True)
test_df.loc[:,'cosine_distance'].fillna(test_df['cosine_distance'].mean(), inplace=True)
test_df.loc[:,'jaccard_distance'].fillna(test_df['jaccard_distance'].mean(), inplace=True)
test_df.loc[:,'braycurtis_distance'].fillna(test_df['braycurtis_distance'].mean(), inplace=True)

f = open('train_F5_tmp_clean.pickle', 'wb') 
pickle.dump(train_df, f)
f.close()

f = open('test_F5_tmp_clean.pickle', 'wb') 
pickle.dump(test_df, f)
f.close()

################################# Join dataframes to complete feature set #####

f = open('train_F4.pickle', 'rb')
train_F4 = pickle.load(f)
f.close()

f = open('test_F4.pickle', 'rb')
test_F4 = pickle.load(f)
f.close()

f = open('train_F5_tmp_clean.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F5_tmp_clean.pickle', 'rb')
test_df = pickle.load(f)
f.close()

new_feats= ['len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2',
            'fuzz_qratio', 'fuzz_WRatio', 'fuzz_token_set_ratio',
            'fuzz_token_sort_ratio', 'wmd', 'norm_wmd', 'cosine_distance',
            'cityblock_distance', 'jaccard_distance', 'canberra_distance',
            'euclidean_distance', 'minkowski_distance', 'braycurtis_distance',
            'skew_q1vec', 'skew_q2vec', 'kur_q1vec', 'kur_q2vec']


train_F4 = train_F4.combine_first(train_df[new_feats])
test_F4 = test_F4.combine_first(test_df[new_feats])

f = open('train_F5.pickle', 'wb') 
pickle.dump(train_F4, f)
f.close()

f = open('test_F5.pickle', 'wb') 
pickle.dump(test_F4, f , protocol=4)
f.close()




