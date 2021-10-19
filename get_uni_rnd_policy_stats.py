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
import cv2
from scipy.ndimage.interpolation import rotate
import copy
import time

class space_carving_rotation():
    def __init__(self, dataset_path,gt_mode=False,rotation_steps=0,total_positions=180):
        self.rotation_steps = rotation_steps #bias of n steps for simulating rotation of the object
        self.total_positions = total_positions #number of posible positions around the circle
        self.masks_files = sorted (glob.glob(os.path.join(dataset_path, 'masks', '*.png')) )#get all .png file names from folder path
        self.extrinsics = self.load_extrinsics(os.path.join(dataset_path, 'extrinsics'))
        #self.bbox = json.load(open(os.path.join(dataset_path, 'bbox.json')))
        self.bbox = json.load(open(os.path.join(dataset_path, '/home/pico/uni/romi/scanner_cube/bbox_min_max.json')))
        self.camera_model = json.load(open(os.path.join(dataset_path, 'camera_model.json')))
        self.intrinsics= self.camera_model['params'][0:4]
        
        params = json.load(open(os.path.join(dataset_path, 'params.json')))

        self.gt_mode = gt_mode

        if self.gt_mode is True:
            self.gt = np.load(os.path.join(dataset_path, 'volumes','vol_180.npy'))
            self.gt_solid_mask = np.where(self.gt==1,True,False) 
            self.gt_n_solid_voxels = np.count_nonzero(self.gt_solid_mask)


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
        self.volume = self.sc.values()

    def carve(self,idx):
        if self.rotation_steps != 0:
            idx = self.calculate_position(idx,-self.rotation_steps)
            
        im = self.load_mask(idx)
        self.space_carve(im, self.extrinsics[idx])
        '''
        if self.rotation_steps == 0:
            self.volume = self.sc.values()
        else:
            self.volume = rotate(self.sc.values(),angle=self.rotation_steps*(360//self.total_positions),reshape=False)'''
        
    def space_carve(self, mask, rt):
        #mask = im.copy() #get_mask(im)
        rot = sum(rt['R'], [])
        tvec = rt['T']
        if self.n_dilation:
            for k in range(self.n_dilation): mask = binary_dilation(mask)    
        self.sc.process_view(self.intrinsics, rot, tvec, mask)


    def gt_compare(self):
        if self.gt_mode is False:
            return 0
        #compare current volume with ground truth (voxelwise) and return percentage
        comp = np.where( self.gt==self.sc.values(),True,False)
        eq_count = np.count_nonzero(comp)
        #perc_sim = (eq_count/np.prod(gt_vol.shape) )*100.
        #perc_sim = (eq_count/682176)*100. #682176number of voxels of the volumes used here 
        perc_sim = eq_count * 0.00014658973637301812
        return perc_sim
    
    def gt_compare_solid(self):
        if self.gt_mode is False:
            return 0
        #compares only solid voxels (with 1;s) between ground truth and test_vol  
        vol_solid_mask = np.where(self.sc.values()==1,True,False) 
        vol_n_solid_voxels = np.count_nonzero(vol_solid_mask)
        intersection = self.gt_solid_mask & vol_solid_mask
        n_intersection = np.count_nonzero(intersection)
        ratio = n_intersection / ( self.gt_n_solid_voxels + vol_n_solid_voxels - n_intersection )
        return ratio 

    def calculate_position(self,init_state,steps):
        n_positions = self.total_positions
        n_pos = init_state + steps
        if n_pos>(n_positions-1):
            n_pos -= n_positions
        elif n_pos<0:
            n_pos += n_positions
        return n_pos




def calculate_position(init_state,steps):
    n_positions = 180
    n_pos = init_state + steps
    if n_pos>(n_positions-1):
        n_pos -= n_positions
    elif n_pos<0:
        n_pos += n_positions
    return n_pos

    

'''def test_random(data_path,n_images,bias):
    spc = space_carving_rotation(data_path,gt_mode=True,rotation_steps=bias)
    pos = np.random.randint(180, size=n_images)
    for i in pos:
        spc.carve(i)
    gt_sim = spc.gt_compare_solid()
    h = np.histogram(spc.volume, bins=3)[0]
    h = h.tolist()
    return gt_sim,h[0],h[1],h[2]'''


'''def test_uniform(data_path,n_images,bias):
    spc = space_carving_rotation(data_path,gt_mode=True,rotation_steps=bias)
    dist = 180//n_images
    for i in range(0,180,dist):
        spc.carve(i)
    gt_sim = spc.gt_compare_solid()
    h = np.histogram(spc.volume, bins=3)[0]
    h = h.tolist()
    return gt_sim,h[0],h[1],h[2]'''

def test_random(data_path,n_images):
    spc = space_carving_rotation(data_path,gt_mode=True,rotation_steps=0)
    pos = np.random.randint(180, size=n_images)
    for i in pos:
        spc.carve(i)
    gt_sim = spc.gt_compare_solid()
    h = np.histogram(spc.volume, bins=3)[0]
    h = h.tolist()
    return gt_sim,h[0],h[1],h[2]


def test_uniform(data_path,n_images,bias):
    spc = space_carving_rotation(data_path,gt_mode=True,rotation_steps=0)
    dist = 180//n_images
    for i in range(0,180,dist):
        spc.carve(calculate_position(i,bias))
    gt_sim = spc.gt_compare_solid()
    h = np.histogram(spc.volume, bins=3)[0]
    h = h.tolist()
    return gt_sim,h[0],h[1],h[2]


def test_all(data_path):
    spc = space_carving_rotation(data_path,gt_mode=True)
    for i in range(180):
        spc.carve(i)       
    h = np.histogram(spc.volume, bins=3)[0]
    h = h.tolist()
    return h[0],h[1],h[2]


dataset_path = '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/for_test'
models =[1,3,25,39,41,6,9,11,14,22,0,20,28,36,48,201,202,203,204,205]
n_images = 5


f = open("uni_rnd_policy_runs_05.json",'w')


data ={}
for m in models:
    m_data = {'rnd':{'gt_dists':[],'empty':[],'uncertain':[],'solid':[],'gt_mean':0,'gt_std':0},'uni':{'gt_dists':[],'empty':[],'uncertain':[],'solid':[],'gt_mean':0,'gt_std':0},'all':{'empty':0,'uncertain':0,'solid':0}}
    plant_path = os.path.join(dataset_path,str(m).zfill(3))
    print(plant_path)
    #collect data random policy
    for i in range(180):
        time1 = time.time()
        gt_dist,empty,uncertain,solid = test_random(plant_path,n_images)
        m_data['rnd']['gt_dists'].append(gt_dist)
        m_data['rnd']['empty'].append(empty)
        m_data['rnd']['uncertain'].append(uncertain)
        m_data['rnd']['solid'].append(solid)
        time2 = time.time()
        print(i, m_data['rnd']['gt_dists'][-1],time2-time1)

    m_data['rnd']['gt_mean'] = float(np.mean(m_data['rnd']['gt_dists']))
    m_data['rnd']['gt_std'] = float(np.std(m_data['rnd']['gt_dists']))

    #collect data uniform distrubution policy
    for i in range(180):
        time1 = time.time()
        gt_dist,empty,uncertain,solid = test_uniform(plant_path,n_images,i)
        m_data['uni']['gt_dists'].append(gt_dist)
        m_data['uni']['empty'].append(empty)
        m_data['uni']['uncertain'].append(uncertain)
        m_data['uni']['solid'].append(solid)
        time2 = time.time()
        print(i, m_data['uni']['gt_dists'][-1],time2-time1)

    m_data['uni']['gt_mean'] = float(np.mean(m_data['uni']['gt_dists']))
    m_data['uni']['gt_std'] = float(np.std(m_data['uni']['gt_dists']))


    empty,uncertain,solid = test_all(plant_path)
    m_data['all']['empty'] = empty
    m_data['all']['uncertain'] = uncertain
    m_data['all']['solid'] = solid


    data[str(m).zfill(3)] = copy.deepcopy(m_data)


json.dump(data, f)
f.close()
    

