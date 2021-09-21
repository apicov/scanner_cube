import open3d as o3d
import cl
import utils as ut
import numpy as np
from skimage.morphology import binary_dilation
import proc3d
import json
from PIL import Image
from utils import *
import glob
import os



class space_carving():
    def __init__(self, dataset_path):
        self.images = self.load_images(os.path.join(dataset_path, 'imgs'))
        self.extrinsics = self.load_extrinsics(os.path.join(dataset_path, 'extrinsics'))
        #self.bbox = json.load(open(os.path.join(dataset_path, 'bbox.json')))
        self.bbox = json.load(open(os.path.join(dataset_path, '/home/pico/uni/romi/scanner_cube/bbox_min_max.json')))
        
        self.camera_model = json.load(open(os.path.join(dataset_path, 'camera_model.json')))
        self.intrinsics= self.camera_model['params'][0:4]
        
        params = json.load(open(os.path.join(dataset_path, 'params.json')))
        self.gt=o3d.io.read_point_cloud(params["gt_path"])
        self.gt_points = np.asarray(self.gt.points)
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
    
    def load_images(self,path):                                                                                                                                         
        imgs = []
        img_files = glob.glob(os.path.join(path, '*.png')) #get all .png files from folder path
        assert len(img_files) != 0,"Image list is empty."
        for i in sorted(img_files):                                                                                                                                     
            img = Image.open(i)                                                                                                                                      
            imgs.append(img.copy())                                                                                                                                     
            img.close()                                                                                                                                                                                                                                                   
        return imgs

    def set_sc(self,bbox):
        x_min, x_max = bbox['x']
        y_min, y_max = bbox['y']
        z_min, z_max = bbox['z']

        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1
        
        print(nx,ny,nz)

        self.origin = np.array([x_min, y_min, z_min])
        self.sc = cl.Backprojection([nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)

    def carve(self,idx):
        self.space_carve(self.images[idx], self.extrinsics[idx])
        
    def space_carve(self, im, rt):
        mask = get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation): mask = binary_dilation(mask)    
        self.sc.process_view(self.intrinsics, rot, tvec, mask)
        
    def dist_to_gt(self):
        vol = self.sc.values().copy()
        vol = vol.reshape(self.sc.shape)
        pcd=proc3d.vol2pcd_exp(vol, self.origin, self.voxel_size, level_set_value=0) 
        pcd_p = np.asarray(pcd.points)
        cd=chamfer_d(self.gt_points , pcd_p)
        return cd 



class space_carving_2():
    def __init__(self, dataset_path):
        self.img_files = sorted (glob.glob(os.path.join(dataset_path, 'imgs', '*.png')) )#get all .png file names from folder path
        self.extrinsics = self.load_extrinsics(os.path.join(dataset_path, 'extrinsics'))
        #self.bbox = json.load(open(os.path.join(dataset_path, 'bbox.json')))
        self.bbox = json.load(open(os.path.join(dataset_path, '/home/pico/uni/romi/scanner_cube/bbox_min_max.json')))
        self.camera_model = json.load(open(os.path.join(dataset_path, 'camera_model.json')))
        self.intrinsics= self.camera_model['params'][0:4]
        
        params = json.load(open(os.path.join(dataset_path, 'params.json')))
        self.gt=o3d.io.read_point_cloud(params["gt_path"])
        self.gt_points = np.asarray(self.gt.points)
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
    
    def load_image(self,idx):                                                                                                                                         
        img = Image.open(self.img_files[idx])                                                                                                                                      
        cp = img.copy()                                                                                                                                 
        img.close()                                                                                                                                                                                                                                                   
        return cp

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
        im = self.load_image(idx)
        self.space_carve(im, self.extrinsics[idx])
        
    def space_carve(self, im, rt):
        mask = get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation): mask = binary_dilation(mask)    
        self.sc.process_view(self.intrinsics, rot, tvec, mask)
        
    def dist_to_gt(self):
        vol = self.sc.values().copy()
        vol = vol.reshape(self.sc.shape)
        pcd=proc3d.vol2pcd_exp(vol, self.origin, self.voxel_size, level_set_value=0) 
        pcd_p = np.asarray(pcd.points)
        cd=chamfer_d(self.gt_points , pcd_p)
        return cd
    
    
    
