import open3d as o3d
import cl
import utils as ut
import numpy as np
from skimage.morphology import binary_dilation
import proc3d
import json
from utils import *
import glob
import os
import csv

class space_carving_2_masks():
    def __init__(self, dataset_path):
        self.masks_files = sorted (glob.glob(os.path.join(dataset_path, 'masks', '*.png')) )#get all .png file names from folder path
        self.extrinsics = self.load_extrinsics(os.path.join(dataset_path, 'extrinsics'))
        #self.bbox = json.load(open(os.path.join(dataset_path, 'bbox.json')))
        self.bbox = json.load(open(os.path.join(dataset_path, '/home/pico/uni/romi/scanner_cube/bbox_min_max.json')))
        self.camera_model = json.load(open(os.path.join(dataset_path, 'camera_model.json')))
        self.intrinsics= self.camera_model['params'][0:4]
        
        params = json.load(open(os.path.join(dataset_path, 'params.json')))
        #self.gt=o3d.io.read_point_cloud(params["gt_path"])
        #self.gt_points = np.asarray(self.gt.points)
        self.n_dilation=params["sc"]["n_dilation"]
        self.voxel_size = params['sc']['voxel_size']
        
        self.set_sc(self.bbox)
        
    def reset(self):
        del(self.sc)
        self.set_sc(self.bbox) 
        
    def load_extrinsics(self,path):
        ext = []
        ext_files = glob.glob(os.path.join(path, '*.json'))
        assert len(ext_files) != 0,"json list is empty."
        for i in sorted(ext_files):                                                                                                                                     
            ext.append(json.load(open(i)))                                                                                                                                                                                                                                                  
        return ext 
    
    def load_mask(self,idx):                                                                                                                                         
        img = cv2.imread(self.masks_files[idx], cv2.IMREAD_GRAYSCALE)                                                                                                                                                                                                                                                                                                                                                                                     
        return img

    def set_sc(self,bbox):
        x_min, x_max = bbox['x']
        y_min, y_max = bbox['y']
        z_min, z_max = bbox['z']

        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1

        self.origin = np.array([x_min, y_min, z_min])
        self.sc = cl.Backprojection([nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)

    def carve(self,idx):
        im = self.load_mask(idx)
        self.space_carve(im, self.extrinsics[idx])
        
    def space_carve(self, mask, rt):
        #mask = im.copy() #get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation): mask = binary_dilation(mask)    
        self.sc.process_view(self.intrinsics, rot, tvec, mask)
        
    '''def dist_to_gt(self):
        vol = self.sc.values().copy()
        vol = vol.reshape(self.sc.shape)
        pcd=proc3d.vol2pcd_exp(vol, self.origin, self.voxel_size, level_set_value=0) 
        pcd_p = np.asarray(pcd.points)
        cd=chamfer_d(self.gt_points , pcd_p)
        return cd'''


parent_dir = "/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/"


for model in range(206):
    data_path = os.path.join( parent_dir,str(model).zfill(3) )
    dest_dir = os.path.join(data_path,'volumes')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(dest_dir)

    spc = space_carving_2_masks(data_path)
    for i in range(180):
        spc.carve(i)
    #h = np.histogram(spc.sc.values(), bins=3)[0]
    
    np.save(os.path.join(dest_dir,'vol_180'), spc.sc.values())
    print("\r{}     ".format(model), end="")

