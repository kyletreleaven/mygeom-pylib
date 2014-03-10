
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

def get_axes3d( fig=None ) :
    if fig is None :
        fig = plt.figure()
        
    ax = fig.add_subplot(111, projection='3d' )
    return ax


