
# standard
import itertools

# scientific
import numpy as np
import networkx as nx

# setiptah
import systems




def box2( r=1., center=None ) :
    DIM = 2
    TWICEDIM = 2*DIM
    
    if center is None : center=np.zeros(DIM)
    
    A = np.zeros( (TWICEDIM,DIM) )
    args = itertools.product( xrange(DIM), [ -1., 1 ] )
    for row, arg in enumerate( args ) :
        col, val = arg
        A[row][col] = val
        
    b = r * np.ones(TWICEDIM) + np.dot( A, center )
    return A, b







""" boundary-of-closed characterization """


def poly2_topology( A, b ) :
    delta = 10**-10     # should obviously make this exact...
    nrow = np.shape(A)[0]
    
    graph = nx.Graph()
    
    edges = { i : [] for i in xrange(nrow) }
    for i,j in itertools.combinations( xrange(nrow), 2 ) :
        AA, bb = systems.system_slice( A, b, [i,j] )
        
        try :
            x = np.linalg.solve( AA, bb )
        except np.linalg.LinAlgError :
            continue
        
        if np.all( np.dot( A, x ) < b + delta ) :
            graph.add_node( (i,j), pos=x )
            edges[i].append( (i,j) )
            edges[j].append( (i,j) )
            
    for i, e in edges.iteritems() :
        for u, v in itertools.combinations( e, 2 ) :    # should be at *most* one
            graph.add_edge( u, v )
            
    return graph




class node : pass



# plotting conversions, convenience

def get_xy( Q ) :
    X = np.array([ x for x,y in Q ])
    Y = np.array([ y for x,y in Q ])
    return X,Y

def vec2(X) :
    x,y = X
    return [0.,x], [0.,y]

def segment(X,Y) :
    x1,y1 = X
    x2,y2 = Y
    return [x1,x2], [y1,y2]


def sample_unitvector() :
    while True :
        U = np.random.rand(3)
        UU = np.linalg.norm( U )
        
        if UU <= 1. : return U/UU


# advanced application-specific plotting convenience

def get_skeleton( topograph, **kwargs ) :
    getpos = lambda u : topograph.node[u].get('pos')
    makeline = lambda u, v : segment( getpos(u), getpos(v) )
    
    from vis import art3d
    lines = [ art3d.Line3D( *makeline(u,v), **kwargs ) for u, v in topograph.edges_iter() ]
    return lines




