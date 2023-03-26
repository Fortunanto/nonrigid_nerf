import os
import json
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import Sampler

from run_nerf_helpers import get_rays
import numpy as np
from torchvision import io

class DummySampler(Sampler):
    def __init__(self, dataset,calibration,N_rand=1024):
        # assert False, f"data {data}"
        self.data = dataset
        self.N_rand=N_rand
        sample_raw_view = [key for key in calibration.keys() if key not in ["focal", "height", "width", "min_bound", "max_bound"]][0]
        self.calibration_raw_view = calibration[sample_raw_view]

    def __iter__(self):
        image_indices = np.random.randint(len(self.data), size=self.N_rand)
        x_coordinates = np.random.randint(self.calibration_raw_view["width"], size=self.N_rand)
        y_coordinates = np.random.randint(self.calibration_raw_view["height"], size=self.N_rand)
        return zip(image_indices,x_coordinates,y_coordinates)
        

    def __len__(self):
        return self.num_samples

class SimulationCollisionDataset(Dataset):
    def __init__(self,dataset_base_fp,dataset_image_fn="images",calibration_fn="calibration.json",image_to_camera_id_and_timestep_fn="image_to_camera_id_and_timestep.json",ray_bending_latent_size=32):
        with open(os.path.join(dataset_base_fp,calibration_fn), "r") as json_file:
            self.calibration = json.load(json_file)
        with open(os.path.join(dataset_base_fp,image_to_camera_id_and_timestep_fn),"r") as json_file:
            self.image_to_camera_id_and_timestep = json.load(json_file)
        self.files = [os.path.join(dataset_base_fp,dataset_image_fn,image) for image in os.listdir(os.path.join(dataset_base_fp,dataset_image_fn))]
        for raw_view in self.calibration.keys():
            # print(raw_view)
            if raw_view in ["focal", "height", "width", "min_bound", "max_bound"]:
                continue
            self.calibration[raw_view]["ray_bending_latent_size"]=ray_bending_latent_size
            # print(raw_view)
    def get_calibration(self):
        return self.calibration
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, key):
        # print(key
        # )
        index,x,y = key
        # assert False, f"index {index}"
        fp = self.files[index]
        fname = fp.split("/")[-1]
        raw_view,timestep = self.image_to_camera_id_and_timestep[fname]
        pose = torch.zeros((3, 5)).to(device="cuda")
        hwf = torch.zeros((3)).to(device="cuda")
        pose[:3,:3] = torch.tensor(self.calibration[raw_view]["rotation"]).to(device=pose.device)
        pose[:3,3] = torch.tensor(self.calibration[raw_view]["translation"]).to(device=pose.device)
        pose[:3,4] = hwf
        rays_o,rays_d=get_rays(pose, self.calibration[raw_view])
        img = io.read_image(fp).to(device=pose.device)
        img = img.permute((1,2,0))
        # assert False, f"rays shape {rays_o.shape,rays_d.shape} img shape {img.shape}"
        rays_rgb = torch.stack((rays_o,rays_d,img),0)[:,y,x,:]
        # assert False, f"shape {rays_rgb.shape}"
        return rays_rgb,(raw_view,timestep)
def get_simulation_collision_dataset(path,batch_size):
    dataset = SimulationCollisionDataset("data/ball_sequence_multiview")
    Dataloader = DataLoader(dataset,batch_size=batch_size,sampler=DummySampler(dataset, dataset.get_calibration(),N_rand=batch_size))
    return Dataloader
# def load_calibration()
if __name__=="__main__":
    batch_size = 1024
    dataset = SimulationCollisionDataset("data/ball_sequence_multiview")
    Dataloader = DataLoader(dataset,batch_size=batch_size,sampler=DummySampler(dataset, dataset.get_calibration(),N_rand=batch_size))
    for rays_rgb,metadata in Dataloader:
        # img,rays = file
        # rays_rgb,metadata = rays_rgb
        print(metadata[1].shape)
        # print(img)
        