
import itertools

import numpy as np
import matplotlib.pyplot as plt


import networkx




def euler_to_quat( yaw, pitch, roll, order='ypr' ) :
    pass

def axial_to_quat( v ) :
    theta = np.linalg.norm( v )
    if theta <= 0. :
      h = np.zeros(4)
      h[0] = 1.
    else :
      u = v / theta    # unit
      t = .5 * theta
      triple = u * np.sin(t)
      h = np.array([ np.cos(t), triple[0], triple[1], triple[2] ])
      
    return h

def conjugate( h ) :
    return np.array([ h[0], -h[1], -h[2], -h[3] ])

def multiply( g, h ) :
    a1,b1,c1,d1 = g
    a2,b2,c2,d2 = h
    a3 = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b3 = a1*b2 + a2*b1 + c1*d2 - c2*d1
    c3 = a1*c2 + a2*c1 + b2*d1 - b1*d2
    d3 = a1*d2 + a2*d1 + b1*c2 - b2*c1
    return np.array([ a3, b3, c3, d3 ])


def rotate_point( p, h ) :
    x,y,z = p
    pp = np.array([ 0., x, y, z ])
    g = conjugate(h)
    qq = multiply( pp, g )
    _, x,y,z = multiply( h, qq )
    return np.array([ x, y, z ])
  
  
def quat_to_matrix( h ) :
    a,b,c,d = h
    
    row1 = [ a*a + b*b, b*c - a*d, b*d + a*c ]
    row2 = [ b*c + a*d, a*a + c*c, c*d - a*b ]
    row3 = [ b*d - a*c, c*d + a*b, a*a + d*d ]
    R = np.array( [ row1, row2, row3 ] )
    
    # some weird post-processing
    R = 2.0*R
    R = R - np.eye(3)
    
    return R


def unit_box(r) :
    A = np.zeros( (6,3) )
    args = itertools.product( xrange(3), [ -1., 1 ] )
    for row, arg in enumerate( args ) :
        col, coeff = arg
        A[row][col] = coeff
        
    b = r * np.ones(6)
    return A, b



def system_slice( A, b, indices ) :
    AA = np.vstack([ A[i,:] for i in indices ])
    bb = np.array([ b[i] for i in indices ])
    return AA, bb

def row_slice( A, indices ) :
    return np.vstack( [ A[i,:] for i in indices ] )

def vector_slice( b, indices ) :
    return np.array( [ b[i] for i in indices ] )

def iter_vertex_tuples( A, b ) :
    """ warning: assumes closed!! """
    nrow = np.shape( A )[0]
    
    for i,j,k in itertools.combinations( xrange(nrow), 3 ) :
        AA,bb = system_slice( A, b, [i,j,k] )
        try :
            x = np.linalg.solve( AA, bb )
        except np.linalg.LinAlgError :
            continue
        
        # yield if feasible
        if np.all( np.dot( A, x ) <= b ) : yield i,j,k
    
def itervertices( A, b ) :
    for i,j,k in iter_vertex_tuples( A, b ) :
        AA, bb = system_slice( A, b, [i,j,k] )
        yield np.linalg.solve( AA, bb )
        
        
        
        
def itertopology( A, b ) :
    delta = 10**-10
    
    nrow = np.shape(A)[0]
    
    for i,j in itertools.combinations( xrange(nrow), 2 ) :
        ai = A[i,:]
        aj = A[j,:]
        U = np.cross(ai,aj)
        UU = np.linalg.norm(U)
        if UU <= delta : continue
        u = U / UU
        
        LE = ( -np.inf, None )
        RE = ( np.inf, None )
        
        Aij = np.vstack([ ai, aj ])
        
        for k in xrange(nrow) :
            if k == i or k == j : continue
            
            ak = A[k,:]
            AA = np.vstack([ Aij, ak ])
            try :
                x = np.linalg.solve( AA, b )
            except np.linalg.LinAlgError :
                continue
            
            




class node : pass



if __name__ == '__main__' :
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    plt.close('all')
    
    # query point
    p = np.array([ 1., 0., 0. ])
    # axis of rotation
    U = np.array([1.,-1.,1.])
    u = U / np.linalg.norm( U ) 
    
    
    # theta
    theta = np.linspace(0,2*np.pi,100)
    
    def proc( t ) :
        h = axial_to_quat( t*u )
        return rotate_point( p, h )
    
    def proc2( t ) :
        h = axial_to_quat( t*u )
        R = quat_to_matrix( h )
        return np.dot(R,p)
    
    def get_xyz( Q ) :
        X = np.array([ x for x,y,z in Q ])
        Y = np.array([ y for x,y,z in Q ])
        Z = np.array([ z for x,y,z in Q ])
        return X,Y,Z
    
    Q1 = [ proc(t) for t in theta ]
    Q2 = [ proc2(t) for t in theta ]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the rotation
    X1,Y1,Z1 = get_xyz(Q1)
    X2,Y2,Z2 = get_xyz(Q2)
    ax.plot(X1,Y1,Z1)
    ax.plot(X2,Y2,Z2+.05)
    
    def vec(X) :
        x,y,z = X
        return [0.,x], [0.,y], [0.,z]
    
    
    # plot the normal vector
    to_plane = np.dot(p,u) * u
    ax.plot( *vec(p) )
    ax.plot( *vec(u) )
    ax.plot( *vec(to_plane) )
    
    ax.set_aspect('equal')
    #plt.plot( theta, X, theta, Y )
    plt.show()
    
    
    
    
    
    
    
  
  
  
