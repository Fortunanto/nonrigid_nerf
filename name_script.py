import argparse
import os 
import re

parser = argparse.ArgumentParser(
                    prog = 'video_from_sim',
                    )
parser.add_argument('data_dir')
args = parser.parse_args()


index = 0
for directory in os.listdir(args.data_dir):
    index = index+1
    for filename in os.listdir(f"{args.data_dir}/{directory}/images"):
        path = f"{args.data_dir}/{directory}/images/{filename}"
        print(f"{directory}/images/{filename}")
    # print(directory)
        filenum = re.findall(r'\d+', f"{filename}")[0]
        # print(filenum)
        os.rename(path,f"{args.data_dir}/camera_{index}_{directory}_{filenum}.png")

