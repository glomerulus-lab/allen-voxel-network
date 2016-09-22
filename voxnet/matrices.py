from .utilities import *
from .mask import *
import numpy as np

def region_laplacian(mask):
    '''
    Generate the laplacian matrix for a given region's voxels. This is the 
    graph laplacian of the neighborhood graph.

    Parameters
    ----------
      mask

    Returns
    -------
      L: num voxel x num voxel laplacian csc_matrix in same order as voxels
        in the region mask
    '''
    import scipy.sparse as sp
    
    voxels=np.array(mask).T
    num_vox=len(mask[0])
    # num_vox=voxels.shape[0]
    vox_lookup={tuple(vox): idx for idx,vox in enumerate(voxels)}
    L=sp.lil_matrix((num_vox,num_vox))
    for idx,vox in enumerate(voxels):
        candidates=possible_neighbors(vox)
        deg=0
        for nei in candidates:
            try:
                idx_nei=vox_lookup[tuple(nei)]
                deg+=1
                L[idx,idx_nei]=1
            except KeyError:
                pass
        L[idx,idx]=-deg
    return L.tocsc()

def gaussian_injection(X,Y,allvox):
    from scipy.optimize import fmin_l_bfgs_b,fmin_bfgs
    from scipy.linalg import cholesky,inv
    # subroutines
    def _vector_gaussian_func(X,A,mu1,mu2,mu3,p11,p12,p13,p22,p23,p33):
        x=X[:,0]
        y=X[:,1]
        z=X[:,2]
        numpt=x.shape[0]
        # D=np.diagflat(np.exp([p11,p22,p33]))
        # # D=np.diagflat([p11,p22,p33])
        # U=np.array([[1,p12,p13],
        #             [0,  1,p23],
        #             [0,  0,  1]],dtype=np.float)
        # pmat=np.dot(U.T,np.dot(D,U))
        U=np.array([[p11,p12,p13],
                    [  0,p22,p23],
                    [  0,  0,p33]],dtype=np.float)
        pmat=np.dot(U.T,U)
        xminusmu=np.array([x-mu1,y-mu2,z-mu3],dtype=np.float)
        return np.exp(A-0.5*np.sum(xminusmu*np.dot(pmat,xminusmu),axis=0))
    # def _vector_gaussian_func(X,A,mu1,mu2,mu3,p11,p12,p13,p22,p23,p33):
    #     ypred=np.zeros((X.shape[0],))
    #     for idx,vox in enumerate(X):
    #         ypred[idx]=_gaussian_func(vox[0],vox[1],vox[2],A,mu1,mu2,mu3,
    #                                   p11,p12,p13,p22,p23,p33)
    #     return ypred
    def _cost_huber(X,Y,A,mu1,mu2,mu3,p11,p12,p13,p22,p23,p33):
        # Define the log-likelihood via the Huber loss function
        def huber_loss(r, c=2):
            t = abs(r)
            flag = t > c
            return np.sum((~flag)*(0.5 * t ** 2)-(flag)*c*(0.5 * c - t),-1)

        ypred=_vector_gaussian_func(X,A,mu1,mu2,mu3,
                                     p11,p12,p13,p22,p23,p33)
        res=Y-ypred
        return float(huber_loss(res))
    # def func(p,X,Y,p0,scale):
    #     pscaled=(p-p0)*scale+p0
    #     mu1,mu2,mu3,p11,p12,p13,p22,p23,p33,A=pscaled
    #     return _cost_huber(X,Y,A,mu1,mu2,mu3,
    #                        p11,p12,p13,p22,p23,p33)
    def func(p,X,Y):
        mu1,mu2,mu3,p11,p12,p13,p22,p23,p33,A=p
        return _cost_huber(X,Y,A,mu1,mu2,mu3,
                           p11,p12,p13,p22,p23,p33)
    # main loop
    Ysum=Y.sum()
    mu=np.sum(X*np.tile(np.reshape(Y,(len(Y),1)),(1,3)),axis=0)/Ysum
    Sigma=np.zeros((3,3),dtype=np.float)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii,jj]=np.sum((X[:,ii]-mu[ii])*(X[:,jj]-mu[jj])*Y,
                                axis=0)/Ysum
    U=cholesky(inv(Sigma))
    print str(U)
    # p0=np.array([mu[0],mu[1],mu[2],
    #              2*np.log(U[0,0]),U[0,1],U[0,2],
    #              2*np.log(U[1,1]),U[1,2],
    #              2*np.log(U[2,2]),
    #              np.log(1)], order='F')
    p0=np.array([mu[0],mu[1],mu[2],
                 U[0,0],U[0,1],U[0,2],
                 U[1,1],U[1,2],
                 U[2,2],
                 np.log(1)], order='F')
    print "initial parameters:\n  %s" % str(p0)
    # scale=np.array([mu[0],mu[1],mu[2],1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1])
    # oput=fmin(func,p0,args=(X,Y,p0,scale),approx_grad=1,disp=1,
    #           factr=1e4,pgtol=1e-8, epsilon=1e-6)
    # oput=fmin_bfgs(func,p0,args=(X,Y),
    #                disp=1,full_output=1,epsilon=1e-8)
    # # gaussian_params_unscaled=oput[0]
    # # gaussian_params=(gaussian_params_unscaled-p0)*scale+p0
    # gaussian_params=oput[0]
    # print "gaussian parameters:\n  %s" % str(gaussian_params)
    # mu1,mu2,mu3,p11,p12,p13,p22,p23,p33,A=gaussian_params
    mu1,mu2,mu3,p11,p12,p13,p22,p23,p33,A=p0
    gaussian_injection=\
      _vector_gaussian_func(allvox,A,mu1,mu2,mu3,
                            p11,p12,p13,p22,p23,p33)
    # renormalize (sets A)
    gaussian_injection=gaussian_injection/gaussian_injection.sum()*Ysum
    return gaussian_injection

def generate_voxel_matrices(mcc,
                            sources, targets, 
                            min_voxels_per_injection = 50,
                            source_coverage          = 0.8,
                            LIMS_id_list             = None,
                            source_shell             = None,
                            laplacian                = None,
                            verbose                  = False,
                            filter_by_ex_target      = True,
                            fit_gaussian             = False,
                            cre                      = False,
                            max_injection_volume     = np.inf,
                            epsilon                  = 0.0):
    '''
    Generates the source and target expression matrices for a set of
    injections, which can then be used to fit the linear model, etc.
    Differs from 'generate_region_matrices' in that they are voxel-resolution,
    i.e. signals are not integrated across regions.

    Parameters
    ----------
    mcc : MouseConnectivityCache
      container object for connectivity data
    sources : list
      source structure ids to consider
    targets : list
      target structure ids to consider
    min_voxels_per_injection : int, default=50
    source_coverage : float, default=0.8
      fraction of injection density that should be contained in union of
      source regions
    LIMS_id_list : list
      experiments to include (default: include all)
    source_shell : int, default=None
      whether to use mask which draws a shell around source regions to
      account for extent of dendrites; integer sets radius in voxels
    laplacian : 'boundary', 'free', or default=None
      Return laplacian matrices? If 'boundary', honor region boundaries.
    verbose : bool, default=False
      print progress
    filter_by_ex_target : bool, default=True
      filter by experiment target region
    fit_gaussian : bool, default=False
      fit a gaussian function to each injection
    cre : bool, default=False
      use Cre injection data?
    max_injection_volume : float, default=np.inf
      filter out experiments with very large injection volumes (mm^3)
        
    Returns
    -------
    experiment_dict, with fields 'experiment_source_matrix',
        'experiment_target_matrix_ipsi', 'experiment_target_matrix_contra',
        'col_label_list_source', 'col_label_list_target', 'row_label_list',
        'source_laplacian', 'Omega',
        'target_laplacian' (if laplacian==True)
    '''

    import scipy.sparse as sp
    from warnings import warn

    def integrate_in_mask(data, query_mask):
        '''
        Integrate a data within a certain query mask.
        Deals with error codes appropriately.

        Parameters
        ----------
        data : 3-array of values
        query_mask : mask over which to integrate (xs, ys, zs)

        Returns
        -------
        sum : integral of data within query_mask
        '''
        def safe_sum(values):
            sum = 0.0
            for val in values:
                if val == -1:
                    val=0.0
                elif val == -2:
                    warn("data error -2, missing tile in LIMS experiment %d" \
                         % curr_LIMS_id)
                    val = 0.0
                elif val == -3:
                    warn("data error -3, no data in LIMS experiment %d" \
                         % curr_LIMS_id)
                    val = 0.0
                sum += val
            return sum
        if mask_len(query_mask) > 0:
            curr_sum = safe_sum(data[query_mask])
        else:
            curr_sum = 0.0
        return curr_sum
    
    def data_in_mask_and_region(data, query_mask, region_mask):
        '''
        Returns the data within a given mask and region.
        Maps the voxel data from a 3d array into a vector.
        Deals with error codes appropriately.

        Parameters
        ----------
        data : 3-array of values
        query_mask : mask over which to integrate (xs, ys, zs)

        '''
        nvox = mask_len(region_mask)
        if mask_len(query_mask)>0:
            data_in_mask = data[region_mask]
            irrelevant_indices = np.ones((nvox,))
            list_relevant = zip(*query_mask)
            list_region = zip(*region_mask)
            for ii,el in enumerate(list_region):
                if el in list_relevant:
                    irrelevant_indices[ii] = 0
            data_in_mask[irrelevant_indices==1] = 0.0
            errmask = (data_in_mask == -1)
            if np.count_nonzero(errmask) > 0:
                data_in_mask[errmask] = 0.0
            errmask = (data_in_mask == -2)
            if np.count_nonzero(errmask) > 0: 
                warn("data error -2, missing tile in LIMS experiment %d" \
                      % curr_LIMS_id)
                data_in_mask[errmask] = 0.0
            errmask = (data_in_mask == -3)
            if np.count_nonzero(errmask) > 0: 
                warn("data error -3, no data in LIMS experiment %d" \
                     % curr_LIMS_id)
                data_in_mask[errmask] = 0.0
        else:
            data_in_mask = np.zeros((nvox,))
        return data_in_mask

    def construct_Omega(injection_mask, region_mask):
        '''
        Construct the Omega matrix entries for a given injection and region.

        Parameters
        ----------
        injection_mask
        region_mask

        Returns
        -------
        omega : vector, same number of voxels as region_mask 
        '''
        nvox = mask_len(region_mask)
        if mask_len(injection_mask) > 0:
            omega = np.ones((nvox,))
            irrelevant_indices = np.ones((nvox,))
            list_relevant = zip(*injection_mask)
            list_region = zip(*region_mask)
            for ii,el in enumerate(list_region):
                if el in list_relevant:
                    irrelevant_indices[ii] = 0
            omega[irrelevant_indices == 1] = 0.0
        else:
            omega=np.zeros((nvox,))
        return omega

    def get_structure_mask_nz(structure_id, ipsi=False, contra=False):
        '''
        Returns the structure mask associated with a given structure id,
        in a sparse format.

        Parameters
        ----------
        structure_id : int
          Id of query structure

        ipsi : bool (default = False)
          Whether to return ipsilateral structure coordinates

        contra : bool (default = False)
          Whether to return contralateral structure coordinates

        If both ipsi == contra == True or False, then return both (default).

        Returns
        -------
        mask_nz : tuple
          Tuple of (x,y,z) coordinates belonging to structure_id
        '''
        if (ipsi and contra) or (not ipsi and not contra):
            return np.where(mcc.get_structure_mask(structure_id)[0])
        elif ipsi and not contra:
            mask = mcc.get_structure_mask(structure_id)
            midline_coord = mask[1]['sizes'][2]/2
            mask_nz = np.where(mask[0])
            idx = (mask_nz[2] >= midline_coord)
            return (mask_nz[0][idx], mask_nz[1][idx], mask_nz[2][idx])
        elif not ipsi and contra:
            mask = mcc.get_structure_mask(structure_id)
            midline_coord = mask[1]['sizes'][2]/2
            mask_nz = np.where(mask[0])
            idx = (mask_nz[2] < midline_coord)
            return (mask_nz[0][idx], mask_nz[1][idx], mask_nz[2][idx])
    
    assert isinstance(source_shell, int) or (source_shell is None),\
      "source_shell should be int or None"

    if verbose:
        print "Creating ExperimentManager and Ontology objects"

    ontology = mcc.get_ontology()

    volume_per_voxel = float(mcc.resolution)**3 * 1e-9

    if verbose:
        print "Creating experiment list"
    if LIMS_id_list is None:
        ex_list = mcc.get_experiments(dataframe=True, cre=cre,
                                      injection_structure_ids=sources.id)
        LIMS_id_list=list(ex_list['id'])
    else:
        ex_list = mcc.get_experiments(dataframe=True, cre=cre,
                                      injection_structure_ids=sources.id)
        ex_list=ex_list[ex_list['id'].isin(LIMS_id_list)]
        LIMS_id_list=list(ex_list['id'])

    # # Determine which experiments should be included:
    # if filter_by_ex_target:
    #     LIMS_id_list_new=[]
    #     ex_list_new=[]
    #     for ii,e in enumerate(ex_list):
    #         # Is the experiment target in one of the source region?
    #         # Note: does not check for leakage into other regions
    #         if e.structure_id in sources:
    #             LIMS_id_list_new.append(LIMS_id_list[ii])
    #             ex_list_new.append(e)
    #     LIMS_id_list=LIMS_id_list_new
    #     ex_list=ex_list_new
    #     del LIMS_id_list_new
    #     del ex_list_new

    mask_len = lambda mask: len(mask[0])
    
    # Get the injection masks
    # if verbose:
    #     print "Getting projection densitiess"
    # ProjD_dict={eid: mcc.get_projection_density(eid) for eid in ex_list.id}
    # if verbose:
    #     print "Getting injection densities"
    # InjD_dict={eid: mcc.get_injection_density(eid) for eid in ex_list.id}
    if verbose:
        print "Computing injection masks from injection_fraction and data_mask"
    injection_mask_dict={eid:
        np.where(np.logical_and(mcc.get_injection_fraction(eid)[0] > epsilon,
                                mcc.get_data_mask(eid)[0]))
                         for eid in ex_list.id}
    
    if source_shell is not None:
        injection_mask_dict_shell = \
          {eid: shell_mask(injection_mask_dict[eid], radius=source_shell)
           for eid in ex_list.id}
        Omega_mask_dict = injection_mask_dict_shell
    else:
        injection_mask_dict_shell={}
        Omega_mask_dict = injection_mask_dict

    # Get the region masks
    if verbose:
        print "Computing region sizes from masks"
    # region_mask_dict = {}
    region_mask_ipsi_dict = {}
    region_mask_contra_dict = {}
    region_nvox = {}
    region_ipsi_nvox = {}
    region_contra_nvox = {}
    nsource = 0 # total voxels in sources
    nsource_ipsi = 0 
    ntarget_ipsi = 0 # total voxels in ipsi targets
    ntarget_contra = 0 # total voxels in contra targets
    source_indices = {}
    source_ipsi_indices = {}
    target_ipsi_indices = {}
    target_contra_indices = {}
    for struct_id in sources.id:
        # region_mask = get_structure_mask_nz(struct_id)
        # region_mask_ipsi_dict[struct_id] = get_structure_mask_nz(struct_id,
        #                                                          ipsi=True)
        # region_mask_contra_dict[struct_id] = get_structure_mask_nz(struct_id,
        #                                                            contra=True)
        #region_mask = np.where(mcc.get_structure_mask(struct_id)[0])
        # region_mask_dict[struct_id] = region_mask
        # idx = (region_mask[2] >= midline_coord)
        # region_mask_ipsi_dict[struct_id] = \
        #   (region_mask[0][idx], region_mask[1][idx], region_mask[2][idx])
        # idx = (region_mask[2] < midline_coord)
        # region_mask_contra_dict[struct_id] = \
        #   (region_mask[0][idx], region_mask[1][idx], region_mask[2][idx])
        region_nvox[struct_id] = mask_len(get_structure_mask_nz(struct_id))
        region_ipsi_nvox[struct_id] = \
          mask_len(get_structure_mask_nz(struct_id, ipsi=True))
        region_contra_nvox[struct_id] = \
          mask_len(get_structure_mask_nz(struct_id, contra=True))
        source_indices[struct_id] = \
          np.arange(nsource, nsource+region_nvox[struct_id])
        source_ipsi_indices[struct_id] = \
          np.arange(nsource_ipsi, nsource_ipsi+region_ipsi_nvox[struct_id])
        nsource += region_nvox[struct_id]
        nsource_ipsi += region_ipsi_nvox[struct_id]
    for struct_id in targets.id:
        # region_mask = get_structure_mask_nz(struct_id)
        # region_mask_ipsi_dict[struct_id] = get_structure_mask_nz(struct_id,
        #                                                          ipsi=True)
        # region_mask_contra_dict[struct_id] = get_structure_mask_nz(struct_id,
        #                                                            contra=True)
        # region_mask = np.where(mcc.get_structure_mask(struct_id)[0])
        # # region_mask_dict[struct_id] = region_mask
        # idx = region_mask[2] >= midline_coord
        # get_structure_mask_nz(struct_id, ipsi=True) = \
        #   (region_mask[0][idx], region_mask[1][idx], region_mask[2][idx])
        # idx = (region_mask[2] < midline_coord)
        # region_mask_contra_dict[struct_id] = \
        #   (region_mask[0][idx], region_mask[1][idx], region_mask[2][idx])
        region_nvox[struct_id] = mask_len(get_structure_mask_nz(struct_id))
        region_ipsi_nvox[struct_id] = \
          mask_len(get_structure_mask_nz(struct_id, ipsi=True))
        region_contra_nvox[struct_id] = \
          mask_len(get_structure_mask_nz(struct_id, contra=True))
        target_ipsi_indices[struct_id] = \
          np.arange(ntarget_ipsi,ntarget_ipsi+region_ipsi_nvox[struct_id])
        target_contra_indices[struct_id] = \
          np.arange(ntarget_contra,ntarget_contra+region_contra_nvox[struct_id])
        ntarget_ipsi += region_ipsi_nvox[struct_id]
        ntarget_contra += region_contra_nvox[struct_id]

    # Compute source mask union
    # TODO: compute iteratively for large instances
    union_of_source_masks = \
      mask_union( *[ get_structure_mask_nz(sid)
                     for sid in sources.id ] )

    # Check for injection mask leaking into other region,
    # restrict to experiments w/o much leakage
    #
    # Also check for too large injection volume.
    LIMS_id_list_new = []
    inj_vols = []
    for LIMS_id in LIMS_id_list:
        inj_mask = injection_mask_dict[LIMS_id]
        # total_pd = integrate_in_mask(mcc.get_projection_density(LIMS_id)[0], inj_mask)
        total_pd = integrate_in_mask(mcc.get_injection_fraction(LIMS_id)[0],
                                     inj_mask)
        # total_source_pd = \
        #   integrate_in_mask(mcc.get_projection_density(LIMS_id)[0],
        #                     mask_intersection(inj_mask, union_of_source_masks))
        total_source_pd = \
          integrate_in_mask(mcc.get_injection_fraction(LIMS_id)[0],
                            mask_intersection(inj_mask, union_of_source_masks))
        source_frac = total_source_pd / total_pd
        injection_volume = total_pd * volume_per_voxel
        inj_vols.append(injection_volume)
        print "  Analyzing experiment %d" % LIMS_id
        print "    source_frac = %f" % source_frac
        print "    injection volume = %f" % injection_volume
        delete_injection = False
        if source_frac < source_coverage:
            print "  Experiment %d has too little coverage" % LIMS_id
            delete_injection = True
        if injection_volume > max_injection_volume:
            print "  Experiment %d has too much injection volume" % LIMS_id
            delete_injection = True
        if delete_injection:
            # del ProjD_dict[LIMS_id]
            del injection_mask_dict[LIMS_id]
            if source_shell is not None:
                del injection_mask_dict_shell[LIMS_id]
        else:
            LIMS_id_list_new.append(LIMS_id)
    inj_vols.sort()
    print "Injection volumes: " + str(inj_vols)
    LIMS_id_list = LIMS_id_list_new
    del LIMS_id_list_new
    num_experiments = len(LIMS_id_list)
    print "Found %d experiments in region" % num_experiments
    if num_experiments < 1:
        raise Exception("number of filtered experiments is zero")
    print "Final list:\n%s" % str(LIMS_id_list)

    # Fit densities with gaussians if required
    if fit_gaussian:
        warn("gaussian fitting not implemented")
    #     ProjD_dict_gaussian={}
    #     for curr_id in LIMS_id_list:
    #         pd=ProjD_dict[curr_id]
    #         # first clean error values
    #         pd[pd==-1]=0.0
    #         where_neg2=np.where(pd==-2)
    #         if np.any(where_neg2):
    #             warn("projection density error -2, missing tile in LIMS experiment %d" % curr_id)
    #         pd[where_neg2]=0.0
    #         where_neg3=np.where(pd==-3)
    #         if np.any(where_neg3):
    #             warn("projection density error -3, no data in LIMS experiment %d" % curr_id)
    #         pd[where_neg3]=0.0
    #         # now fit gaussian injections
    #         X=np.where(pd)
    #         Y=pd[X]
    #         print "Fitting gaussian for experiment %d" % curr_id
    #         new_pd_vec=gaussian_injection(np.array(X).T,Y)
    #         new_pd=np.zeros(pd.shape)
    #         new_pd[X]=new_pd_vec
    #         ProjD_dict_gaussian[curr_id]=new_pd
    
    # Initialize matrices:
    structures_above_threshold_ind_list = []
    experiment_source_matrix_pre = np.zeros((len(LIMS_id_list), nsource_ipsi))
    Omega = np.zeros((len(LIMS_id_list), nsource_ipsi))
    col_label_list_source = np.zeros((nsource_ipsi, 1))
    voxel_coords_source = np.zeros((nsource_ipsi, 3))

    # Source :
    if verbose:
        print "Getting source densities"
    for jj, struct_id in enumerate(sources.id):
        # Get the region mask:
        curr_region_mask = get_structure_mask_nz(struct_id, ipsi=True)
        ipsi_injection_volume_list = []
        for ii, curr_LIMS_id in enumerate(LIMS_id_list):
            # Get the injection mask:
            # Below is commented because we don't count the shell voxels
            # if source_shell is not None:
            #     curr_experiment_mask=injection_mask_dict_shell[curr_LIMS_id]
            # else:
            curr_experiment_mask = injection_mask_dict[curr_LIMS_id]
            # Compute density, source:
            intersection_mask = mask_intersection(curr_experiment_mask, 
                                                  curr_region_mask)
            if mask_len(intersection_mask) > 0:
                # if fit_gaussian:
                #     pd_at_intersect=data_in_mask_and_region(
                #         InjD_dict_gaussian[curr_LIMS_id][0],
                #         intersection_mask,
                #         curr_region_mask)
                # else:
                indices = source_ipsi_indices[struct_id]
                ipsi_injection_volume_list.append(mask_len(intersection_mask))
                col_label_list_source[indices] = struct_id
                these_coords = np.array(curr_region_mask).T
                voxel_coords_source[indices,] = these_coords
                experiment_source_matrix_pre[ii,indices] = \
                  data_in_mask_and_region(
                      mcc.get_injection_density(curr_LIMS_id)[0],
                      intersection_mask, curr_region_mask
                      )
                Omega[ii,indices] = \
                  construct_Omega(
                      mask_intersection(Omega_mask_dict[curr_LIMS_id],
                                        curr_region_mask),
                      curr_region_mask
                      )
        
        # Determine if current structure should be included in source list:
        ipsi_injection_volume_array = np.array(ipsi_injection_volume_list)
        num_exp_above_thresh = len(np.nonzero(ipsi_injection_volume_array
                                              >= min_voxels_per_injection)[0])
        if num_exp_above_thresh > 0:
            structures_above_threshold_ind_list.append(jj)
            if verbose:
                print("structure %s above threshold") % struct_id
    
    Omega = sp.csc_matrix(Omega)

    # if len(structures_above_threshold_ind_list) < len(sources):
    #     raise Exception('length of structures_above_threshold_ind_list < sources')
    # # restrict matrices to the good experiments & structures
    # above_threshold_indices=
    # experiment_source_matrix=experiment_source_matrix_pre[:,structures_above_threshold_ind_list]
    experiment_source_matrix = experiment_source_matrix_pre
    row_label_list = np.array(LIMS_id_list)
     
    # Target:
    if verbose:
        print "Getting target densities"
    experiment_target_matrix_ipsi = np.zeros((len(LIMS_id_list),
                                              ntarget_ipsi))
    experiment_target_matrix_contra = np.zeros((len(LIMS_id_list), 
                                                ntarget_contra))
    col_label_list_target_ipsi = np.zeros((ntarget_ipsi,1))
    col_label_list_target_contra = np.zeros((ntarget_contra,1))
    voxel_coords_target_ipsi = np.zeros((ntarget_ipsi,3))
    voxel_coords_target_contra = np.zeros((ntarget_contra,3))
    for jj, struct_id in enumerate(targets.id):
        # Get the region mask:
        curr_region_mask_ipsi = get_structure_mask_nz(struct_id, ipsi=True)
        curr_region_mask_contra = get_structure_mask_nz(struct_id, contra=True)
        for ii, curr_LIMS_id in enumerate(row_label_list):
            # Get the injection mask:
            if source_shell is not None:
                curr_experiment_mask = injection_mask_dict_shell[curr_LIMS_id]
            else:
                curr_experiment_mask = injection_mask_dict[curr_LIMS_id]
            # Compute integrated density, target, ipsi:
            difference_mask = \
              mask_difference(curr_region_mask_ipsi,curr_experiment_mask)
            indices_ipsi = target_ipsi_indices[struct_id]
            pd_at_diff = \
              data_in_mask_and_region(
                  mcc.get_projection_density(curr_LIMS_id)[0],
                  difference_mask, curr_region_mask_ipsi
                  )
            experiment_target_matrix_ipsi[ii, indices_ipsi] = pd_at_diff
            col_label_list_target_ipsi[indices_ipsi] = struct_id
            voxel_coords_target_ipsi[indices_ipsi,] = \
              np.array(curr_region_mask_ipsi).T
            # Compute integrated density, target, contra:    
            difference_mask = \
              mask_difference(curr_region_mask_contra, curr_experiment_mask)
            indices_contra = target_contra_indices[struct_id]
            pd_at_diff = \
              data_in_mask_and_region(
                  mcc.get_projection_density(curr_LIMS_id)[0],
                  difference_mask, curr_region_mask_contra)
            experiment_target_matrix_contra[ii, indices_contra] = pd_at_diff
            col_label_list_target_contra[indices_contra] = struct_id
            voxel_coords_target_contra[indices_contra,] = \
              np.array(curr_region_mask_contra).T

    if verbose:
        print "Getting laplacians"
    # Laplacians
    if laplacian == 'boundary':
        Lx = \
          sp.block_diag(tuple(
              [region_laplacian(get_structure_mask_nz(region, ipsi=True))
               for region in sources.id]
              ))
        Ly_ipsi = \
          sp.block_diag(tuple(
              [region_laplacian(get_structure_mask_nz(region, ipsi=True))
               for region in targets.id]
              ))
        Ly_contra = \
          sp.block_diag(tuple(
              [region_laplacian(get_structure_mask_nz(region, contra=True))
               for region in targets.id]
              ))
    elif laplacian == 'free':
        #m = mask_union(*[region_mask_ipsi_dict[sid] for sid in targets.id])
        m = np.hstack(tuple([get_structure_mask_nz(region, ipsi=True)
                             for region in sources.id]))
        Lx = region_laplacian(m)
        m = np.hstack(tuple([get_structure_mask_nz(region, ipsi=True)
                             for region in targets.id]))
        Ly_ipsi = region_laplacian(m)
        m = np.hstack(tuple([get_structure_mask_nz(region, contra=True)
                             for region in targets.id]))
        Ly_contra = region_laplacian(m)
    Lx=sp.csc_matrix(Lx)
    Ly_ipsi=sp.csc_matrix(Ly_ipsi)
    Ly_contra=sp.csc_matrix(Ly_contra)
    if verbose:
        print "Done."

    # Include only structures with sufficient injection information, and 
    # experiments with one nonzero entry in row:
    experiment_dict={}
    experiment_dict['experiment_source_matrix']=experiment_source_matrix
    experiment_dict['experiment_target_matrix_ipsi']=\
      experiment_target_matrix_ipsi
    experiment_dict['experiment_target_matrix_contra']=\
      experiment_target_matrix_contra
    experiment_dict['col_label_list_source']=col_label_list_source 
    experiment_dict['col_label_list_target_ipsi']=col_label_list_target_ipsi
    experiment_dict['col_label_list_target_contra']=col_label_list_target_contra
    experiment_dict['row_label_list']=row_label_list
    experiment_dict['voxel_coords_source']=voxel_coords_source
    experiment_dict['voxel_coords_target_ipsi']=voxel_coords_target_ipsi
    experiment_dict['voxel_coords_target_contra']=voxel_coords_target_contra
    experiment_dict['Omega']=Omega
    if laplacian:
        experiment_dict['Lx']=Lx
        experiment_dict['Ly_ipsi']=Ly_ipsi
        experiment_dict['Ly_contra']=Ly_contra
    
    return experiment_dict

def generate_region_matrices(data_dir,
                             source_id_list, target_id_list, 
                             min_voxels_per_injection = 50,
                             LIMS_id_list = None, source_shell=False):
    '''
    Generates the source and target expression matrices for a set of
    injections, which can then be used to fit the linear model, etc.

    Modified from mesoscale_connectivity_linear_model by Nicholas Cain

    Parameters
    ----------
      data_dir: directory containing hdf5 files
      source_id_list: source structure ids to consider
      target_id_list: target structure ids to consider
      min_voxels_per_injection, default=50
      LIMS_id_list: list of experiments to include
      source_shell, default=None:
        whether to use mask which draws a shell
        around source regions to account for extent of dendrites,
        set to an integer
        
    Returns
    -------
      experiment_dict, with fields 'experiment_source_matrix',
        'experiment_target_matrix_ipsi', 'experiment_target_matrix_contra',
        'col_label_list_source', 'col_label_list_target', 'row_label_list'
    '''
    from friday_harbor.structure import Ontology
    from friday_harbor.mask import Mask
    from friday_harbor.experiment import ExperimentManager
    from warnings import warn
    
    
    EM=ExperimentManager(data_dir=data_dir)
    ontology=Ontology(data_dir=data_dir)

    if LIMS_id_list == None:
        ex_list=EM.all()
        LIMS_id_list=[e.id for e in ex_list]
    else:
        ex_list=[EM.experiment_by_id(LIMS_id) for LIMS_id in LIMS_id_list]
    
    # Get the injection masks
    PD_dict={}
    injection_mask_dict={}
    injection_mask_dict_shell={}
    for e in ex_list:
        curr_LIMS_id=e.id
        PD_dict[curr_LIMS_id]=e.density()
        injection_mask_dict[curr_LIMS_id]=e.injection_mask()
        if source_shell == True:
            injection_mask_dict_shell[curr_LIMS_id]=e.injection_mask(shell=True)
    
    # Get the region masks
    region_mask_dict={}
    region_mask_ipsi_dict={}
    region_mask_contra_dict={}
    for curr_structure_id in source_id_list + target_id_list:
        region_mask_dict[curr_structure_id]=ontology.get_mask_from_id_nonzero(curr_structure_id)
        region_mask_ipsi_dict[curr_structure_id]=ontology.get_mask_from_id_right_hemisphere_nonzero(curr_structure_id)
        region_mask_contra_dict[curr_structure_id]=ontology.get_mask_from_id_left_hemisphere_nonzero(curr_structure_id)

    def integrate_in_mask(curr_LIMS_id, intersection_mask):
        def safe_sum(values):
            sum=0.0
            for val in values:
                if val == -1:
                    val=0.0
                elif val == -2:
                    warn(('projection density error -2, ',
                          'missing tile in LIMS experiment %d')
                                  % curr_LIMS_id)
                    val=0.0
                elif val == -3:
                    warn(('projection density error -3, ',
                          'no data in LIMS experiment %d')
                                  % curr_LIMS_id)
                    val=0.0
                sum+=val
            return sum
        if len(intersection_mask) > 0:
            curr_sum=safe_sum(PD_dict[curr_LIMS_id][intersection_mask.mask])
        else:
            curr_sum=0.0
        return curr_sum
    
    # Initialize matrices:
    structures_above_threshold_ind_list=[]
    experiment_source_matrix_pre=np.zeros((len(LIMS_id_list), 
                                             len(source_id_list)))
    
    # Source:
    for jj, curr_structure_id in enumerate(source_id_list):
        # Get the region mask:
        curr_region_mask=region_mask_dict[curr_structure_id]
        ipsi_injection_volume_list=[]
        for ii, curr_LIMS_id in enumerate(LIMS_id_list):
            # Get the injection mask:
            curr_experiment_mask=injection_mask_dict[curr_LIMS_id]
            # Compute integrated density, source:
            intersection_mask=Mask.intersection(curr_experiment_mask, curr_region_mask)
            experiment_source_matrix_pre[ii, jj]=integrate_in_mask(curr_LIMS_id, intersection_mask)
            ipsi_injection_volume_list.append(len(intersection_mask)) 
    
        # Determine if current structure should be included in source list:
        ipsi_injection_volume_array=np.array(ipsi_injection_volume_list)
        num_exp_above_thresh=len(np.nonzero(ipsi_injection_volume_array 
                                            >= min_voxels_per_injection)[0])
        if num_exp_above_thresh > 0:
            structures_above_threshold_ind_list.append(jj)
            
    # Determine which experiments should be included:
    expermiments_with_one_nonzero_structure_list=[] 
    for ii, row in enumerate(experiment_source_matrix_pre):
        if row.sum() > 0.0:
            expermiments_with_one_nonzero_structure_list.append(ii)

    if len(structures_above_threshold_ind_list) < len(source_id_list):
        raise Exception('length of structures_above_threshold_ind_list < source_id_list')
    
    experiment_source_matrix=experiment_source_matrix_pre[:,structures_above_threshold_ind_list][expermiments_with_one_nonzero_structure_list,:]
    row_label_list=np.array(LIMS_id_list)[expermiments_with_one_nonzero_structure_list]
    col_label_list_source=np.array(source_id_list)[structures_above_threshold_ind_list]
     
    # Target:
    experiment_target_matrix_ipsi=np.zeros((len(row_label_list), 
                                            len(target_id_list)))
    experiment_target_matrix_contra=np.zeros((len(row_label_list), 
                                              len(target_id_list)))
    for jj, curr_structure_id in enumerate(target_id_list):
        # Get the region mask:
        curr_region_mask_ipsi=region_mask_ipsi_dict[curr_structure_id]
        curr_region_mask_contra=region_mask_contra_dict[curr_structure_id]
        
        for ii, curr_LIMS_id in enumerate(row_label_list):
            # Get the injection mask:
            if source_shell == True:
                curr_experiment_mask=injection_mask_dict_shell[curr_LIMS_id]
            else:
                curr_experiment_mask=injection_mask_dict[curr_LIMS_id]
            # Compute integrated density, target, ipsi:
            difference_mask=curr_region_mask_ipsi.difference(curr_experiment_mask)
            experiment_target_matrix_ipsi[ii, jj]=integrate_in_mask(curr_LIMS_id, 
                                                                    difference_mask)
            
            # Compute integrated density, target, contra:    
            difference_mask=curr_region_mask_contra.difference(curr_experiment_mask)
            experiment_target_matrix_contra[ii, jj]=integrate_in_mask(curr_LIMS_id, difference_mask) 

    # Include only structures with sufficient injection information, and 
    # experiments with one nonzero entry in row:
    experiment_dict={}
    experiment_dict['experiment_source_matrix']=experiment_source_matrix
    experiment_dict['experiment_target_matrix_ipsi']=experiment_target_matrix_ipsi
    experiment_dict['experiment_target_matrix_contra']=experiment_target_matrix_contra
    experiment_dict['col_label_list_source']=col_label_list_source 
    experiment_dict['col_label_list_target']=np.array(target_id_list)
    experiment_dict['row_label_list']=row_label_list
    experiment_dict['W0_ipsi']=np.zeros(experiment_source_matrix.shape[1],
                                    experiment_target_matrix_ipsi.shape[1])
    experiment_dict['W0_contra']=np.zeros(experiment_source_matrix.shape[1],
                                    experiment_target_matrix_contra.shape[1])
    return experiment_dict
