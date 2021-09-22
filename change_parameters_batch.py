import numpy as np
import json
#import tensorflow as tf
import os
import time
import glob

parent_dir = "/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/"

dirs = sorted ( glob.glob(os.path.join(parent_dir, '*')) )
d_file = 'params.json'

for d in dirs:
    params = json.load(open(os.path.join(d,d_file )))
    params['sc']['voxel_size'] = 0.6
    print(params)

    with open(os.path.join(d, d_file ), 'w') as json_file:
        json.dump(params, json_file)
    
