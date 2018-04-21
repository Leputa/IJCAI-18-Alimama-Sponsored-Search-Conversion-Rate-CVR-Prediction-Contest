import numpy as np
import os
from scipy.sparse import csr_matrix

PATH = '../Cache/'

def save_sparse_csr(filename, array):
    filename = os.path.join(PATH, filename)
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    filename = os.path.join(PATH,filename)
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])