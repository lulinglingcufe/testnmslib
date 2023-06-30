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


all_data_matrix = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_base.fvecs") 
all_data_matrix_query = fvecs_read("/home/ubuntu/lulingling/testnmslib/sift/sift_query.fvecs")  


#query_matrix = all_data_matrix[0:1000]
query_matrix = all_data_matrix_query[0:100]
data_matrix = all_data_matrix[0:500000]  #[0:550000]












# query_matrix = all_data_matrix[0:600]
# data_matrix = all_data_matrix[600:10000]
# Create a held-out query data set
# (data_matrix, query_matrix) = train_test_split(all_data_matrix, test_size = 0.1)

# print("# of queries %d, # of data points %d"  % (query_matrix.shape[0], data_matrix.shape[0]) )

# Set index parameters
# These are the most important onese
#M = 6
M = 8
efC = 100

num_threads = 1
#index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 2}
#没有用skip_optimized_index参数，应该就是默认使用优化方法。

print('Index-time parameters', index_time_params)

# Number of neighbors 
K=10
# Space name should correspond to the space name 
# used for brute-force search
space_name='l2'# l2
# Intitialize the library, specify the space, the type of the vector and add data points 
index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR) 
index.addDataPointBatch(data_matrix) 



# Create an index
start = time.time()
index.createIndex(index_time_params) 
end = time.time() 
print('Index-time parameters', index_time_params)
print('Indexing time = %f' % (end-start))

# Setting query-time parameters
#efS = 40 #10万数据的时候
#efS = 50 #20万数据的时候
#efS = 60 #40万数据的时候
#efS = 70 #60万数据的时候
#efS = 80 #80万数据的时候
#efS = 90 #100万数据的时候
efS = 50


# query_time_params = {'efSearch': efS}
# print('Setting query-time parameters', query_time_params)
# index.setQueryTimeParams(query_time_params)


# # Querying 查询计时
# query_qty = query_matrix.shape[0]
# start = time.time() 
# nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
# end = time.time() 
# print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
#       (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 
