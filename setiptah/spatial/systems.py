
import itertools

import numpy as np
import matplotlib.pyplot as plt


import networkx as nx




def system_slice( A, b, indices ) :
    AA = np.vstack([ A[i,:] for i in indices ])
    bb = np.array([ b[i] for i in indices ])
    return AA, bb

def row_slice( A, indices ) :
    return np.vstack( [ A[i,:] for i in indices ] )

def vector_slice( b, indices ) :
    return np.array( [ b[i] for i in indices ] )




