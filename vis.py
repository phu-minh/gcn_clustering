#import pandas as pd
import numpy as np
#'/Users/minhphu/Work/kltn/dgl-0.0/examples/pytorch/hilander/data/subcenter_arcface_deepglint_train_1_in_10_recreated.pkl'
knn = np.load('features/knn.graph.512.bf.npy')
#print(knn[0])
features = np.load('features/512.fea.npy')
labels = np.load ('features/512.labels.npy')
print(features.ndim)
print(labels.ndim)
print(labels.shape)
print(labels)

print('-------------------')
knn = np.load('features/train_knn.npy',allow_pickle=True)
#print(knn[0])
features = np.load('features/train_features.npy',allow_pickle=True)
labels = np.load ('features/train_labels.npy',allow_pickle=True)
print(features.ndim)
print(labels.ndim)
print(labels.shape)
print(labels)