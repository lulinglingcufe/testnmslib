import h5py

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

#with h5py.File('/home/ubuntu/lulingling/testnmslib/glove-200-angular.hdf5', 'r') as file:
with h5py.File('/home/ubuntu/lulingling/testnmslib/glove-25-angular.hdf5', 'r') as file:
    all_data_matrix  = file['train']#train test distances neighbors
    all_data_matrix_query = file['test']


    query_matrix = all_data_matrix_query[0:100]
    data_matrix = all_data_matrix[0:550000]  #200000   5000



M = 8
efC = 100

num_threads = 1
#index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 2}
#没有用skip_optimized_index参数，应该就是默认使用优化方法。

print('Index-time parameters', index_time_params)


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
