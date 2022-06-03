'''
create_2d_matrices.py

Takes 2d flattened data and forms it into matrices for use in the regression
problem.
'''

import os
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.structure_tree import StructureTree
import numpy as np
import h5py
import time
import nrrd
import matplotlib.pyplot as plt
from scipy.io import mmwrite
from voxnet.conn2d import *
from voxnet.utilities import h5write
import scipy.sparse as sp
from scipy.io import savemat
from mcmodels.core.cortical_map import CorticalMap

view_str = 'top_view'
drive_path = os.path.join('../mouse_connectivity_models/connectivity')
output_dir = os.path.join('../create_matrice_output/'+view_str)

no_data = -1
#no_data = 0
mapper = CorticalMap(projection='top_view')

# When downloading 3D connectivity data volumes, what resolution do you want
# (in microns)?  
# Options are: 10, 25, 50, 100
resolution_um=100

# Downsampling factor
stride = 1

# Omega threshold
Omega_thresh = 0.80

# The manifest file is a simple JSON file that keeps track of all of
# the data that has already been downloaded onto the hard drives.
# If you supply a relative path, it is assumed to be relative to your
# current working directory.
manifest_file = os.path.join(drive_path, "mcmodels_manifest.json")


# Start processing data
mcc = MouseConnectivityCache(manifest_file=manifest_file,
                             resolution=resolution_um)

# Injection structure of interest
isocortex = mcc.get_structure_tree().get_structures_by_name(['Isocortex'])

# Open up a pandas dataframe of all of the experiments
experiments = mcc.get_experiments(dataframe=True, 
                                  injection_structure_ids=\
                                      [isocortex[0]['id']], 
                                  cre=False)
print("%d total experiments" % len(experiments))

# Load look up tables from mapper
view_lut = mapper.view_lookup[:]
view_paths = mapper.paths[:]

# Compute masks
view_lut = downsample(view_lut, stride)
data_mask = np.where(view_lut != no_data)

# Right indices
right = np.zeros(view_lut.shape, dtype=bool)
right[:, int(view_lut.shape[1]/2):] = True

# Right hemisphere data
hemi_R_mask = np.where(np.logical_and(view_lut != no_data, right))
# Left hemisphere data
hemi_L_mask = np.where(np.logical_and(view_lut != no_data,
                                      np.logical_not(right)))

# Laplacians
Lx = laplacian_2d(hemi_R_mask)
Ly = laplacian_2d(data_mask)
mmwrite(os.path.join(output_dir, "Lx.mtx"), Lx)
mmwrite(os.path.join(output_dir, "Ly.mtx"), Ly)

# Initialize matrices to be filled in
nx = len(hemi_R_mask[0])
ny = len(data_mask[0])
X = np.zeros((nx, len(experiments)))
Y = np.zeros((ny, len(experiments)))
Omega = np.zeros((ny, len(experiments)))

t0 = time.time()
index = 0

for eid, row in experiments.iterrows():
    print("\nRow %d\nProcessing experiment %d" % (index,eid))
    print(row)
    # Get experiment data
    data_path = "../original_0.8/allen_sdk_experiments_topview/"
    data_dir = "experiment_%d/" % eid
    data_dir = os.path.join(data_path, data_dir)
    
    # Get and remap injection data
    in_fn = data_dir + "injection_density_" + view_str + "_%d.nrrd" \
      % int(resolution_um)
    
    print("reading " + in_fn)
    in_d_s = downsample(nrrd.read(in_fn)[0], stride)
    
    # Get and remap projection data
    pr_fn = data_dir + "projection_density_" + view_str + "_%d.nrrd" \
      % int(resolution_um)
    
    print("reading " + pr_fn)
    pr_d_s = downsample(nrrd.read(pr_fn)[0], stride)

    # fill matrices
    X[:, index] = in_d_s[hemi_R_mask]
    Y[:, index] = pr_d_s[data_mask]
    
    # Attempt to get unique Omega_thresh for each experiment to avoid
    # selecting too little values or too small of values
    #vector = in_d_s[np.logical_and(in_d_s != 0, data_mask != 0)]
    #Omega_thresh = np.median(vector)
    
    this_Omega = (in_d_s[data_mask] > Omega_thresh).astype(int)
    Omega[:, index] = this_Omega
    index += 1

t1 = time.time()
total = t1 - t0
print("%0.1f minutes elapsed" % (total/60.))

# Save Matrices
voxel_coords_source = np.array(hemi_R_mask).T
voxel_coords_target = np.array(data_mask).T
Omega = sp.csc_matrix(Omega)

h5write(os.path.join(output_dir, "X.h5"), X)
h5write(os.path.join(output_dir, "Y.h5"), Y)
mmwrite(os.path.join(output_dir, "Omega.mtx"), Omega)

#savemat(os.path.join(output_dir, 'create_view_lut.mat'), {'view_lut' : view_lut})
       
savemat(os.path.join(output_dir, 'matrices.mat'),
        {'X': X, 'Y': Y, 'Lx': Lx, 'Ly': Ly, 'Omega': Omega,
         'voxel_coords_source': voxel_coords_source,
         'voxel_coords_target': voxel_coords_target,
         'view_lut': view_lut, 'stride': stride, 'Omega_thresh': Omega_thresh},
        oned_as='column', do_compression=True)
