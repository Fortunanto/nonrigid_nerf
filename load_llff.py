import numpy as np
import os

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1./(bds.max() - bds.min())
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test

def load_llff_data_old(datadir, factor, recenter, bd_factor, spherify):

    import json

    with open("./data/example_sequence/precomputed.json", "r") as json_file:
        precomputed = json.load(json_file)
    poses = np.array(precomputed["poses"])
    bds = np.array(precomputed["bds"])
    render_poses = np.array(precomputed["render_poses"])
    i_test = precomputed["i_test"]

    import imageio

    images = sorted(os.listdir("./data/example_sequence/images/"))
    images = (
        np.stack(
            [
                imageio.imread(
                    os.path.join("./data/example_sequence/images/", image),
                    ignoregamma=True,
                )
                for image in images
            ],
            axis=-1,
        )
        / 255.0
    )
    images = np.moveaxis(images, -1, 0).astype(np.float32)

    return images, poses, bds, render_poses, i_test

def load_llff_data_multi_view(datadir, factor, recenter, bd_factor, spherify):
   
    import imageio
    images = sorted(os.listdir(os.path.join(datadir, "images")))

    images = (
        np.stack(
            [
                imageio.imread(
                    os.path.join(datadir, "images", image),
                    ignoregamma=True,
                )
                for image in images
            ],
            axis=-1,
        )
        / 255.0
    )

    images = np.moveaxis(images, -1, 0).astype(np.float32)

    from train import _get_multi_view_helper_mappings
    extras = _get_multi_view_helper_mappings(len(images), datadir)

    import json
    with open(os.path.join(datadir, "calibration.json"), "r") as json_file:
        calibration = json.load(json_file)
    poses = np.zeros((len(images), 3, 5))
    hwf = np.array([0., 0., 0.])
    for i in range(poses.shape[0]):

        raw_view = extras["raw_views"][extras["imageid_to_viewid"][i]]

        poses[i,:3,:3] = np.array(calibration[raw_view]["rotation"])
        poses[i,:3,3] = np.array(calibration[raw_view]["translation"])
        poses[i,:3,4] = hwf

    bds = np.array([calibration["min_bound"], calibration["max_bound"]])
    
    render_poses = poses.copy() # dummy
    i_test = 0 # dummy

    return images, poses, bds, render_poses, i_test
