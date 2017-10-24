# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:51:53 2017

@author: Reg
"""

import pandas as pd
from igraph import Graph
import dask
from dask.diagnostics import ProgressBar
import dask.bag as db
import pickle

train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

train_orig.fillna(' ', inplace=True)
test_orig.fillna(' ', inplace=True)

g = Graph()
g.add_vertices(train_orig.question1)
g.add_vertices(train_orig.question2)
g.add_vertices(test_orig.question1)
g.add_vertices(test_orig.question2)

g.add_edges(train_orig[['question1', 'question2']].to_records(index=False))
g.add_edges(test_orig[['question1', 'question2']].to_records(index=False))

def graph_feat(row, g, n=3):
    
    q1_in = row[0]
    q2_in = row[1]
    H = g.subgraph(sum(g.neighborhood([q1_in, q2_in], 3),[]))
    H.radius()
    
    q1_node = H.vs.find(name=q1_in)
    q2_node = H.vs.find(name=q2_in)
    
    out = []
    
    #H.vertex_disjoint_paths(q1_node.index, q2_node.index)
    
    #evcent
    tmp = H.evcent()
    out.append(tmp[q1_node.index])
    out.append(tmp[q2_node.index])
    out.append((tmp[q1_node.index]+tmp[q2_node.index])/2)
    out.append(abs(tmp[q1_node.index]-tmp[q2_node.index]))
    
    # edge disjoint paths
    out.append(H.edge_disjoint_paths(q1_node.index, q2_node.index))
    
    # pagerank
    out += [sum(H.pagerank([q1_node.index, q2_node.index]))/2]
    
    # transitivity
    out.append(H.transitivity_avglocal_undirected())
    tmp = H.transitivity_local_undirected([q1_node.index, q2_node.index])
    out.append(tmp[0])
    out.append(tmp[1])
    out.append(abs(tmp[0]-tmp[1]))
    out.append(H.transitivity_undirected())
    
    #assortativity
    out.append(H.assortativity_degree())
    out.append(H.average_path_length())
    
    tmp = H.betweenness([q1_node.index, q2_node.index])
    out.append(tmp[0])
    out.append(tmp[1])
    out.append((tmp[0]+tmp[1])/2)
    out.append(abs(tmp[0]-tmp[1]))
    
    
    out.append(H.bibcoupling(q1_node.index)[0][q2_node.index])
    out.append(H.density())
    
    tmp1 = H.eccentricity(q1_node.index)
    tmp2 = H.eccentricity(q2_node.index)
    avg  = (tmp1+tmp2)/2
    diff = abs(tmp1-tmp2)
    
    out.append(tmp1)
    out.append(tmp2)
    out.append(avg)
    out.append(diff)
    
    out.append(H.girth())
    
    out.append(H.maxflow_value(q1_node.index, q2_node.index))
    
    out.append(H.similarity_inverse_log_weighted(q1_node.index)[0][q2_node.index])
    
    out.append(H.similarity_jaccard(pairs=[(q1_node.index, q2_node.index)])[0])


    return out


col_names = ['evcent1', 'evcent2','evcent12' ,'evcent_diff',
             'edp','pgrnk',
             'trns_avglocal', 'trns1','trns2','trns12','trns_diff',
             'ass_deg','avg_pathlength',
             'between1','between2','between12','between_diff',
             'bibcoup','density',
             'ecc1', 'ecc2', 'ecc_avg', 'ecc_diff',
             'girth', 'maxflow', 'simm_log', 'simm_jacc']

# Do dask computation
# Do train set



dask.set_options(get=dask.multiprocessing.get)
tmp = train_orig[['question1', 'question2']].values.tolist()

b = db.from_sequence(tmp, npartitions=4)

with ProgressBar():
    data_out = b.map(lambda x: graph_feat(x, g, n=3)).compute(get=dask.multiprocessing.get)


f = open('train_F7_raw.pickle', 'wb') 
pickle.dump(data_out, f)
f.close()

# Do test set

dask.set_options(get=dask.multiprocessing.get)
tmp = test_orig[test_orig['question1']!=test_orig['question2']][['question1', 'question2']].values.tolist()

b = db.from_sequence(tmp, npartitions=4)

with ProgressBar():
    data_out = b.map(lambda x: graph_feat(x, g, n=3)).compute(get=dask.multiprocessing.get)

f = open('test_F7_raw.pickle', 'wb') 
pickle.dump(data_out, f)
f.close()
















