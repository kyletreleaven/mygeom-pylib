
import numpy as np

# for LP
from cvxopt import matrix
import cvxopt.solvers as solvers
solvers.options['show_progress'] = False

#from pulp import *


def system_slice( A, b, indices ) :
    AA = np.vstack([ A[i,:] for i in indices ])
    bb = np.array([ b[i] for i in indices ])
    return AA, bb

def system_omit( A, b, i ) :
    nrow = np.shape( A )[0]
    indices = set( xrange(nrow) )
    indices.remove(i)
    AA, bb = system_slice( A, b, indices )
    return AA, bb




def is_redundant( A, b, i ) :
    ai = matrix( A[i,:] )
    bi = b[i]
    #
    nrow = np.shape( A )[0]
    indices = set( xrange(nrow) )
    remain = indices.difference( [i] )
    AA, bb = system_slice( A, b, remain )
    AA = matrix(AA)
    bb = matrix(bb)
    
    soln = solvers.lp( -ai, AA, bb )        # -ai to maximize
    #print soln
    
    # check for violation
    if soln['status'] == 'optimal' :
        y = -soln['primal objective']
        if y <= bi : return True
        
    return False


def find_redundant_constraints( A, b ) :
    redundant = set()
    
    nrow = np.shape( A )[0]
    indices = set( xrange(nrow) )
    for i in indices :
        if is_redundant( A, b, i ) : redundant.add( i )
        
    return redundant


def eliminate_redundant_constraints( A, b ) :
    nrow = np.shape( A )[0]
    indices = set( xrange(nrow) )
    redundant = find_redundant_constraints( A, b )
    
    AA, bb = system_slice( A, b, indices.difference( redundant ) )
    return AA, bb






if __name__ == '__main__' :
    import geom3
    from vis import *
    
    plt.close('all')
    
    A1,b1 = geom3.box( 1. )
    A2,b2 = geom3.box( 2. )
    
    A = np.vstack([ A1, A2 ])
    b = np.hstack([ b1, b2 ])
    
    AA, bb = eliminate_redundant_constraints( A, b )
    g = geom3.polytope_topology( AA, bb )
    lines = geom3.get_skeleton( g, color='b' )
    
    
    # show to people
    ax = get_axes3d()
    ax.set_xlim3d(-10,10)
    ax.set_ylim3d(-10,10)
    ax.set_zlim3d(-10,10)

    for line in lines : ax.add_line( line )
    ax.set_aspect('equal')
    plt.show()
    
    










