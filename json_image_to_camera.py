import json
import argparse
import os 
import re

parser = argparse.ArgumentParser(
                    prog = 'video_from_sim',
                    )
parser.add_argument('data_dir')
parser.add_argument('output_dir')
args = parser.parse_args()

image_to_camera_id_and_timestep = {}
# image_to_camera_id_and_timestep['a']=(1,2)
# image_to_camera_id_and_timestep['b']=(1,2,4,6)

for image in os.listdir(args.data_dir):
    if image=="filtered":
        continue
    name = "_".join(image.split('_')[2:5])
    timestep = int(image.split("_")[-1].split(".")[0])
    if timestep%6==1:
        image_to_camera_id_and_timestep[image]=(name,int(timestep/6))
        # os.rename(f"{args.data_dir}/{image}", f"{args.data_dir}/filtered/{image}")

# assert False, f"image_to_camera_id_and_timestep {image_to_camera_id_and_timestep}"
with open(f"{args.output_dir}/image_to_camera_id_and_timestep.json", "w") as json_file:
    json.dump(image_to_camera_id_and_timestep, json_file, indent=4)

