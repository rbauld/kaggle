# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import argparse
import functools
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter

def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
    
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True) #1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True) #2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True) #3

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True) #4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True) #5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True) #6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True) #7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True) #8

    f = functools.partial(wc_diff_unique_stop, stops=stops)    
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)    
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) #10

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True) #11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True) #12

    f = functools.partial(char_diff_unique_stop, stops=stops) 
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #13

#     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  #15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  #16
    
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True) #17    

    return X


parser = argparse.ArgumentParser(description='XGB with Handcrafted Features')
parser.add_argument('--save', type=str, default='XGB_leaky',
                    help='save_file_names')
args = parser.parse_args()

df_train = pd.read_csv('./input/train_features.csv', encoding="ISO-8859-1")
X_train_ab = df_train.iloc[:, 2:-1]
X_train_ab = X_train_ab.drop('euclidean_distance', axis=1)
X_train_ab = X_train_ab.drop('jaccard_distance', axis=1)

df_train = pd.read_csv('./input/train.csv')
df_train = df_train.fillna(' ')

df_test = pd.read_csv('./input/test.csv')
del df_test


# explore
stops = set(stopwords.words("english"))

df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

words = [x for y in train_qs for x in y]
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Building Features')
X_train = build_features(df_train, stops, weights)
X_train = pd.concat((X_train, X_train_ab), axis=1)

train_df = X_train
train_df.fillna(train_df.mean(), inplace=True)

# Test features
print('Building Test Features')
df_test = pd.read_csv('./input/test_features.csv', encoding="ISO-8859-1")
x_test_ab = df_test.iloc[:, 2:-1]
x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)
df_test = pd.read_csv('./input/test.csv')
df_test = df_test.fillna(' ')

df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

x_test = build_features(df_test, stops, weights)
x_test = pd.concat((x_test, x_test_ab), axis=1)


test_df = x_test
test_df.fillna(train_df.mean(), inplace=True)

del x_test, X_train, X_train_ab, counts, df_test, df_train, train_qs, weights, x_test_ab
del words

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(train_df.mean(), inplace=True)

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.fillna(train_df.mean(), inplace=True)

dupe_cols = ['word_match', 'len_q1', 'len_q2', 'diff_len', 'fuzz_qratio', 
             'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
             'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio'
             'wmd', 'norm_wmd', 'cosine_distance', 'cityblock_distance', 'canberra_distance',
             'minkowski_distance', 'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
             'fuzz_token_sort_ratio', 'wmd', 'jaccard']

unique_cols = [x for x in train_df.columns if x not in dupe_cols]

train_df = train_df[unique_cols]
test_df  = test_df[unique_cols]

import pickle
f = open('train_F7.pickle', 'rb')
train_df2 = pickle.load(f)
f.close()

f = open('test_F7.pickle', 'rb')
test_df2 = pickle.load(f)
f.close()


train_df2 = train_df2.combine_first(train_df)
test_df2 = test_df2.combine_first(test_df)

del train_df, test_df

 
f = open('train_F8.pickle', 'wb') 
pickle.dump(train_df2, f)
f.close()

f = open('test_F8.pickle', 'wb') 
pickle.dump(test_df2, f , protocol=4)
f.close()