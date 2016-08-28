function [coord_vox_map_source,...
          coord_vox_map_target_ipsi,...
          coord_vox_map_target_contra]=lookups(voxel_coords_source, ...
                                               voxel_coords_target_contra,...
                                               voxel_coords_target_ipsi)
    coord_vox_map_source=index_lookup_map(voxel_coords_source);
    coord_vox_map_target_contra= ...
        index_lookup_map(voxel_coords_target_ipsi);
    coord_vox_map_target_ipsi= ...
        index_lookup_map(voxel_coords_target_contra);
end
