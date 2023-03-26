import os
import json
import torch
from torch.utils.data import Dataset
from run_nerf_helpers import get_rays_np

class SimulationCollisionDataset(Dataset):
    def __init__(self,dataset_base_fp,dataset_image_fn="images",calibration_fn="calibration.json",image_to_camera_id_and_timestep_fn="image_to_camera_id_and_timestep.json",ray_bending_latent_size=32):
        with open(os.path.join(dataset_base_fp,calibration_fn), "r") as json_file:
            self.calibration = json.load(json_file)
        with open(os.path.join(dataset_base_fp,image_to_camera_id_and_timestep_fn),"r") as json_file:
            self.image_to_camera_id_and_timestep = json.load(json_file)
        self.files = [os.path.join(dataset_base_fp,image) for image in os.listdir(os.path.join(dataset_base_fp,dataset_image_fn))]
        for raw_view in self.calibration.keys():
            # print(raw_view)
            if raw_view in ["focal", "height", "width", "min_bound", "max_bound"]:
                continue
            self.calibration[raw_view]["ray_bending_latent_size"]=ray_bending_latent_size
            # print(raw_view)
        print(self.calibration)

        assert False
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, index):
        fp = self.files[index]
        fname = fp.split("/")[-1]
        rays = np.stack([get_rays_np(p, intrinsics[dataset_extras["imageid_to_viewid"][imageid]]) for imageid, p in enumerate(poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]

        print(self.image_to_camera_id_and_timestep[fname])
        return self.files[index]

# def load_calibration()
for file in SimulationCollisionDataset("data/ball_sequence_multiview"):
    print(file)