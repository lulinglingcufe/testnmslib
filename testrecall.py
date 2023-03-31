import logging
logging.basicConfig(level=logging.NOTSET)

import numpy 
import sys 
import nmslib 
import time 
import math 
# from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import train_test_split
print(sys.version)
print("NMSLIB version:", nmslib.__version__)

# Just read the data
#all_data_matrix = numpy.loadtxt('/home/ubuntu/lulingling/testnmslib/final128_10K.txt')
# all_data_matrix = numpy.random.randn(1000, 128).astype(numpy.float32)


def ivecs_read(fname):
    a = numpy.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


all_data_matrix = ivecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_groundtruth.ivecs") 


query_matrix = all_data_matrix[0:1000]
#data_matrix = all_data_matrix[1000:1000000]



print('Index-time parameters', query_matrix)
