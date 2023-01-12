# -*- coding: utf-8 -*-

# import tensorflow as tf
import pickle
import os
import random
import hashlib
from hierarchicalNetwork import HierarchicalNetwork

def build_hiera(item):
    top = item['root']
    tokens = item['tokens']
    
    return build_hiera_rec(tokens, top)

def build_hiera_rec(tokens, top):
    excep = ['\n']
    
    top_new = []
    for i in tokens[top[1]]['children_index']:
        if not tokens[i]['text'] in excep:
            top_new.append(build_hiera_rec(tokens,[tokens[i]['text'],i]))
    if len(top_new) == 0:
        return top
    else:
        return [top,top_new]
    
def prt_hiera(li):
    print(li)
    res, res_width = prt_hiera_rec(li,0)
    for i in res:
        print(i)

def prt_hiera_rec(li, deep):
    if not isinstance(li[1], list):
        return [str(li)], len(str(li))+10
    l_new = [str(li[0])]
    l = []
    width = 0
    new_res_width = 0
    for i in li[1]:
        res, res_width = prt_hiera_rec(i,deep + 1)
        for i in range(len(res)):
            res[i] = res[i].ljust(res_width)
        l , width = prt_hiera_merge_str(l,width,res,res_width)
        new_res_width = new_res_width + res_width
    return l_new + l, new_res_width + 1

def prt_hiera_merge_str(li,w1,li_new,w2):
    t1 = li
    t2 = li_new
    res = []
    while len(t1) > 0 or len(t2) > 0:
        if len(t1) == 0:
            for i in t2:
                res.append(''.ljust(w1) + i)
            return res, w1+w2
        elif len(t2) == 0:
            for i in t1:
                res.append(i + ''.ljust(w2))
            return res, w1+w2
        else:
            res.append(t1[0].ljust(w1)  + t2[0].ljust(w2))
            t1 = t1[1:]
            t2 = t2[1:]
    return res, w1+w2

def create_cv_batches(s, cv_num = 10, randomize = False):
    samples = s.copy()
    n = len(samples)
    if randomize:
        random.shuffle(samples)
    for i in range(cv_num):
        x_train = []
        x_test = []
        for j in range(cv_num):
            if i==j:
                x_test.extend(range(n*j//cv_num,n*(j+1)//cv_num))
            else:
                x_train.extend(range(n*j//cv_num,n*(j+1)//cv_num))
    with open('log.txt', 'w') as f:
        print(x_train, x_test, file=f)
    yield (x_train,x_test)

def main():
    cwd = os.getcwd()
    
    with open(cwd + '/data/wikisent_1000_parsing_embedding.pkl', 'rb') as f:
        parsing_embedding = pickle.load(f)
    with open(cwd + '/data/wikisent_1000_embedding_centroids.pkl', 'rb') as f:
        centroids = pickle.load(f)
        
    print(parsing_embedding[0].keys())
    print(parsing_embedding[0]['tokens'][0].keys())
    
    hiera = {}
    wordIdxMap = {}
    IdxWordMap = {}
    idxtos = {}
    idx = 0
    idx2 = 0
    for sentence in parsing_embedding:
        dic = hashlib.md5(sentence['text'].encode('utf-8')).hexdigest()
        hiera[dic] = build_hiera(sentence)
        idxtos[idx2] = sentence
        idx2 = idx2 + 1
        for term in sentence['tokens']:
            if not term['text'] in wordIdxMap and not term['text'] in ['\n']:
                wordIdxMap[term['text']] = idx
                IdxWordMap[idx] = term['text']
                idx = idx + 1
                
    model = HierarchicalNetwork(centroids,wordIdxMap,IdxWordMap,hiera,idxtos)
    
    
    
#     for j in range(len(parsing_embedding)):
#         parsing_embedding[j].pop('tokens')
#         parsing_embedding[j].pop('root')
#         for i in range(len(parsing_embedding[j]['tokens'])):    
#             parsing_embedding[j]['tokens'][i].pop('embedding')

    for (x_train, x_valid) in create_cv_batches(parsing_embedding):
        model.train(x_train, [1.0]*len(x_train))

if __name__ == '__main__':
    main()