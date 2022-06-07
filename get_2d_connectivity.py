'''
get_2d_connectivity.py

Fetches connectivity data and maps it into 2d, saving the results.
'''

import os
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.structure_tree import StructureTree
import numpy as np
import h5py
import time
import nrrd
import ipdb
import matplotlib.pyplot as plt
from scipy.io import savemat
from mcmodels.core.cortical_map import CorticalMap

view_str = 'top_view'
drive_path = os.path.join('../mouse_connectivity_models/connectivity')
output_dir = os.path.join(os.getenv('HOME'), 'allen_get_connectivity_output')
mapper = CorticalMap(projection='top_view')

#Quick fix (removes holes in top_view)
mapper.view_lookup[51, 69] = mapper.view_lookup[51, 68]
mapper.view_lookup[51, 44] = mapper.view_lookup[51, 43]

no_data = -1
#no_data = 0

# When downloading 3D connectivity data volumes, what resolution do you want
# (in microns)?  
# Options are: 10, 25, 50, 100
resolution_um = 100

# Drop list criterion, in percent difference
volume_fraction = 20

# The manifest file is a simple JSON file that keeps track of all of
# the data that has already been downloaded onto the hard drives.
# If you supply a relative path, it is assumed to be relative to your
# current working directory.
manifest_file = os.path.join(drive_path, "mcmodels_manifest.json")

# Start processing data
mcc = MouseConnectivityCache(manifest_file = manifest_file, resolution=resolution_um)
                             
# Injection structure of interest
isocortex = mcc.get_structure_tree().get_structures_by_name(['Isocortex'])
isocortex_id = isocortex[0]['id']

# Open up a pandas dataframe of all of the experiments
experiments = mcc.get_experiments(dataframe=True, 
                                  injection_structure_ids=\
                                      [isocortex[0]['id']], 
                                  cre=False)
print("%d total experiments" % len(experiments))

# Get lookup table from mapper
view_lut = mapper.view_lookup[:]
view_paths = mapper.paths[:]

# Compute size of each path to convert path averages to sums
norm_lut = np.zeros(view_lut.shape, dtype=int)
ind = np.where(view_lut != no_data)
ind = zip(ind[0], ind[1])
for curr_ind in ind:
    curr_path_id = view_lut[curr_ind]
    curr_path = view_paths[curr_path_id, :]
    norm_lut[curr_ind] = np.sum(curr_path != no_data)

t0 = time.time()
expt_drop_list = []
full_vols = []
flat_vols = []

# Map injections and projections to 2D space
for eid, row in experiments.iterrows():
    print("\nProcessing experiment %d" % eid)
    print(row)
    
    # Create path for experiment data to be saved
    new_path = "../allen_sdk_experiments_topview"
    if(os.path.exists(new_path) == False):
        os.mkdir(new_path)
    new_dir = "experiment_%d" % eid
    data_dir = os.path.join(new_path, new_dir)
    if(os.path.exists(data_dir) == False):
        os.mkdir(data_dir)
        
    # get and remap injection data
    print("getting injection density")
    in_d, in_info = mcc.get_injection_density(eid)
    
    print("mapping to surface")
    in_d_s  = mapper.transform(in_d,fill_value = np.nan)
        
    flat_vol = np.nansum(in_d_s * norm_lut) * (10./1000.)**3
    flat_vols.append(flat_vol)
    full_vol = np.nansum(in_d) * (10./1000.)**3
    full_vols.append(full_vol)
    print("flat_vol = %f\nfull_vol = %f" % (flat_vol, full_vol))
    
    # drop experiments without much injection volume
    if np.abs(flat_vol - full_vol) / full_vol * 100 > volume_fraction:
        print ("warning, placing experiment in drop list")
        expt_drop_list.append(eid)
            
    in_fn = data_dir + "/injection_density_" + view_str + "_%d.nrrd" \
      % int(resolution_um)
    
    print("writing " + in_fn)
    nrrd.write(in_fn, in_d_s)
    
    # get and remap projection data
    print ("getting projection density")
    pr_d, pr_info = mcc.get_projection_density(eid)
    
    print ("mapping to surface")
    pr_d_s  = mapper.transform(pr_d, fill_value = np.nan)
    
    pr_fn = data_dir + "/projection_density_" + view_str + "_%d.nrrd" \
      % int(resolution_um)
    
    print ("writing " + pr_fn)
    nrrd.write(pr_fn, pr_d_s)

t1 = time.time()
total = t1 - t0
print ("%0.1f minutes elapsed" % (total/60.))
print ("flat vols: " + str(flat_vols))
print ("full vols: " + str(full_vols))
print ("drop list: " + str(expt_drop_list))

#savemat(os.path.join(output_dir, view_str +'_get_conn_view_lut.mat'), {'view_lut' : view_lut})

savemat(os.path.join(output_dir, view_str + '_volumes.mat'),
        {'flat_vols': flat_vols, 'full_vols': full_vols,
         'drop_list': expt_drop_list,
         'experiment_ids': np.array(experiments['id'])},
        oned_as='column', do_compression=True)


# import matplotlib.pyplot as plt
# plt.ion()
# fig = plt.figure(figsize = (10,10))
# ax = fig.add_subplot(121)
# h = ax.imshow(in_d_s)
# #fig.colorbar(h)

# #fig2 = plt.figure(figsize = (10,10))
# ax2 = fig.add_subplot(122)
# h2 = ax2.imshow(pr_d_s)
# #fig2.colorbar(h2)


