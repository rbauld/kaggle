# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

ques = pd.concat([train_orig[['question1', 'question2']], \
        test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')
ques.shape

q_dict = defaultdict(set)

for i in range(len(ques)):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])
        
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

import pickle
f = open('train_F5.pickle', 'rb')
train_df = pickle.load(f)
f.close()

f = open('test_F5.pickle', 'rb')
test_df = pickle.load(f)
f.close()


train_df['q1_q2_intersect'] = train_orig['q1_q2_intersect'] 
test_df['q1_q2_intersect'] = test_orig['q1_q2_intersect'] 

        
import pickle
f = open('train_F6.pickle', 'wb') 
pickle.dump(train_df, f)
f.close()

f = open('test_F6.pickle', 'wb') 
pickle.dump(test_df, f , protocol=4)
f.close()



