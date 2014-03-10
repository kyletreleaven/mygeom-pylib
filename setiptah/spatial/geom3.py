
import itertools

import numpy as np
import matplotlib.pyplot as plt


import networkx as nx




def euler_to_quat( yaw, pitch, roll, order='ypr' ) :
    raise NotImplementedError()

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


def box( r=1., center=None ) :
    if center is None : center=np.zeros(3)
    
    A = np.zeros( (6,3) )
    args = itertools.product( xrange(3), [ -1., 1 ] )
    for row, arg in enumerate( args ) :
        col, coeff = arg
        A[row][col] = coeff
        
    b = r * np.ones(6) + np.dot( A, center )
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
        
        # find a point on the line
        Aij, bij = system_slice( A, b, [i,j] )
        A0 = np.vstack( [ Aij, u ] )
        b0 = np.append( bij, 0. )
        x0 = np.linalg.solve( A0, b0 )
        
        LE = ( -np.inf, [] )
        RE = ( np.inf, [] )
        for k in xrange(nrow) :
            if k == i or k == j : continue
            
            ak = A[k,:]
            bk = b[k]
            
            # row k may completely eliminate the line... e.g.,
            # ...if the plane is parallel to the line and line is on the wrong side
            # step 1. see if line is parallel
            if np.abs( np.dot( ak, u ) ) <= delta :
                # step 2. see if a point on the line is on the wrong side
                if np.dot(ak,x0) > bk :
                    LE = ( np.inf, None )
                    RE = ( -np.inf, None )
                    
            # if not, try to update the intercepts
            else :
                triple = tuple(sorted( (i,j,k) ))
                AA, bb = system_slice( A, b, triple )
                try :
                    x = np.linalg.solve( AA, bb )
                except np.linalg.LinAlgError :
                    # wholly contained... safe.
                    continue
                
                # find projection of x onto line of u
                y = np.dot( x, u )
                
                if np.dot( ak, u ) > 0. :
                    if y < RE[0] :
                        RE = ( y, [ triple ] )
                    elif y <= RE[0] :
                        RE[1].append( triple )
                
                elif np.dot( ak, u ) < 0. :
                    if y > LE[0] :
                        LE = ( y, [ triple ] )
                    elif y >= LE[0] :
                        LE[1].append( triple )
                        
                else:
                    raise 'nope!'
                    
            # if the segment has vanished, no need to proceed?
            vanished = ( LE[0] >= RE[0] )
            if vanished : break
            
        # if the segment vanished, don't report
        if vanished : continue
        # otherwise LE and RE have left and right ends to them... should be finite
        assert LE[0] > -np.inf and RE[0] < np.inf
        
        # until we handle degenerate cases
        _, le = LE
        _, re = RE
        
        yield le[0], re[0]


def polytope_topology( A, b ) :
    res = nx.Graph()
    
    for u, v in itertopology( A, b ) :
        res.add_edge( u, v )
        
    for u, data in res.nodes_iter( data=True ) :
        i,j,k = u
        AA, bb = system_slice( A, b, [i,j,k] )
        data['pos'] = np.linalg.solve( AA, bb )
        
    return res




class node : pass



# plotting conversions, convenience

def get_xyz( Q ) :
    X = np.array([ x for x,y,z in Q ])
    Y = np.array([ y for x,y,z in Q ])
    Z = np.array([ z for x,y,z in Q ])
    return X,Y,Z

def vec(X) :
    x,y,z = X
    return [0.,x], [0.,y], [0.,z]

def segment(X,Y) :
    x1,y1,z1 = X
    x2,y2,z2 = Y
    return [x1,x2], [y1,y2], [z1,z2]


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







if __name__ == '__main__' :
    from vis import *
    
    plt.close('all')
    
    
    if False :
        # query point
        p = np.array([ 1., 0., 0. ])
        # axis of rotation
        U = np.array([1.,-1.,1.])
        # make sure it's normal!
        u = U / np.linalg.norm( U ) 
        
        # theta
        theta = np.linspace(0,3./2*np.pi,100)
        
        def proc( t ) :
            h = axial_to_quat( t*u )
            return rotate_point( p, h )
        
        def proc2( t ) :
            h = axial_to_quat( t*u )
            R = quat_to_matrix( h )
            return np.dot(R,p)
        
        Q1 = [ proc(t) for t in theta ]
        Q2 = [ proc2(t) for t in theta ]
        
        # visualize
        fig = plt.figure()
        ax = get_axes3d( fig )
        
        # plot the rotation
        X1,Y1,Z1 = get_xyz(Q1)
        X2,Y2,Z2 = get_xyz(Q2)
        ax.plot(X1,Y1,Z1)
        ax.plot(X2,Y2,Z2+.05)
        
        # plot the normal vector
        to_plane = np.dot(p,u) * u
        ax.plot( *vec(p) )
        ax.plot( *vec(u) )
        ax.plot( *vec(to_plane) )
        ax.set_aspect('equal')
        
        plt.plot( theta, X, theta, Y )
        plt.show()
        
        
        
        
    if True :
        # make a basic box
        A, b = box( 2. )
        
        # make an extra slice
        u = sample_unitvector()
        k = 2. * np.random.rand() + 1.
        k = .5 * k
        #print u, k
        A = np.vstack( [ A, u ] )
        b = np.append( b, k )
        
        # shift the box
        x0 = np.array([ 5., 0, 0 ])
        b += np.dot( A, x0 )
        
        # obtain the shape's topology
        g1 = polytope_topology( A, b )
        
        # axis of rotation
        U = np.array([0.,0.,1.])
        u = U / np.linalg.norm( U )     # make sure it's normal!
        
        # rotate by pi/2
        h = axial_to_quat( .75 * np.pi * u )
        Rstar = quat_to_matrix( conjugate( h ) )
        A2 = np.dot( A, Rstar )
        b2 = b
        
        # obtain rotated topology
        g2 = polytope_topology( A2, b2 )
        
        
        
        # prepare stage
        fig = plt.figure()
        ax = get_axes3d( fig )
        ax.set_xlim3d(-10,10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(-10,10)
        
        lines1 = get_skeleton( g1, color='b' )
        lines2 = get_skeleton( g2, color='g' )
        for line in lines1 + lines2 : ax.add_line( line )
        #linecoll = art3d.Line3DCollection( lines )
        #ax.add_collection3d( linecoll )
        
        # plot the normal vector
        rot_plane_through_center = np.dot(x0,u) * u
        #ax.plot( *vec(p) )
        ax.plot( *vec(u) )
        ax.plot( *vec(rot_plane_through_center) )
        ax.plot( *vec(x0) )
        ax.plot( *vec( rotate_point(x0,h) ) )
        
        # show to people
        ax.set_aspect('equal')
        plt.show()
    
    
    
    
    
    
    
    
    
    
  
  
  
