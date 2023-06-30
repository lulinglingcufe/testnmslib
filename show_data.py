import h5py

import logging
logging.basicConfig(level=logging.NOTSET)

import sys 
#import nmslib 
import time 
import math 

import logging
logging.basicConfig(level=logging.NOTSET)

import numpy 


# from sklearn.neighbors import NearestNeighbors (打印fvecs)
# from sklearn.model_selection import train_test_split
print(sys.version)

# def ivecs_read(fname):
#     a = numpy.fromfile(fname, dtype='int32')
#     d = a[0]
#     return a.reshape(-1, d + 1)[:, 1:].copy()


# def fvecs_read(fname):
#     return ivecs_read(fname).view('float32')


# all_data_matrix = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_base.fvecs") 
# #all_data_matrix_query = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_query.fvecs")  

# query_matrix = all_data_matrix[0]
# print('query_matrix', query_matrix)



######################################（打印hdf5里的data 向量）
with h5py.File('/home/ubuntu/lulingling/testnmslib/glove-25-angular.hdf5', 'r') as file:
    all_data_matrix  = file['train']#train test distances neighbors
    all_data_matrix_query = file['test']
    query_matrix = all_data_matrix[0]
    print('query_matrix', query_matrix)