from plyfile import PlyData
import numpy as np

def read_ply(file_path):
    plydata = PlyData.read(file_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    points = np.array([x, y, z]).T
    
    if 'red' in plydata['vertex'] and 'green' in plydata['vertex'] and 'blue' in plydata['vertex']:
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']
        colors = np.array([r, g, b]).T
    else:
        colors = None 
    
    return points, colors
