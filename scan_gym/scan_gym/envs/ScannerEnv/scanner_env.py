import numpy as np
#import cv2
from os import listdir
from os.path import isfile, join
import gym
from gym import error, spaces, utils
import glob
from PIL import Image
import open3d as o3d
from .cl import *
from skimage.morphology import binary_dilation
from .proc3d import *
import json
from .utils import *
import glob
import os
from .space_carving import *

MODELS_PATH = '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/'


class ScannerEnv(gym.Env):
    """
    A template to implement custom OpenAI Gym environments
    """
    metadata = {'render.modes': ['human']}
    def __init__(self,dataset_path,init_pos_inc_rst=False):
        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        self.dataset_path = dataset_path
        self.n_positions = 180 #total of posible positions in env
        self.init_pos_inc_rst = init_pos_inc_rst #if false init position is random, if true, it starts in position 0 and increments by 1 position every reset
        self.init_pos_counter = 0
        
        self.volume_shape = (66,68,152)
        self.im_ob_space = gym.spaces.Box(low=-1, high=1, shape=self.volume_shape, dtype=np.float32)

        #current position                                          
        #lowl = np.array([0])
        #highl = np.array([179])                                           
        #self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.float32)
        self.vec_ob_space  = spaces.Discrete(self.n_positions)


        self.observation_space = gym.spaces.Tuple((self.im_ob_space, self.vec_ob_space))



        self.actions = {0:90,1:3,2:5,3:11,4:23,5:45,6:-45,7:-23,8:-11,9:-5,10:-3,11:-90}
        self.action_space = gym.spaces.Discrete(12)

        #self._spec.id = "Romi-v0"
        self.reset()

    def reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.done = False
        self.kept_abs_images = [] #real position of images in dataset
        self.kept_rel_images = [] #relative position used in environment 
                
        self.current_position = 0
        
        if self.init_pos_inc_rst : #position bias increases at every reset (like rotating plant in space)
            if self.init_pos_counter >= self.n_positions:
                self.init_pos_counter = 0
            self.position_bias = self.init_pos_counter
            self.init_pos_counter += 1
        else:
            self.position_bias =  np.random.randint(0,self.n_positions)


        #the image at the beginning position is always kept
        self.kept_rel_images.append(0) #(self.current_position)
        self.absolute_position = self.position_bias #( self.calculate_position(self.current_position,self.position_bias) )
        self.kept_abs_images.append(self.absolute_position) 
        

        if self.dataset_path == '':
            model = np.random.randint(10) #we use first 10 models from database for training
            self.spc = space_carving_2_masks( os.path.join(MODELS_PATH,str(model).zfill(3)) )
        else:
            self.spc = space_carving_2_masks(self.dataset_path)

            
        self.spc.carve(self.absolute_position) 
        self.current_state = ( self.spc.sc.values() , 0) #self.spc.sc.values().astype('float16')

        #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
        h = np.histogram(self.spc.sc.values(), bins=3)[0]
        self.last_vspaces_count = h[0]   #spaces count from last sd volume carving
        
        return self.current_state


    @property
    def nA(self):
        return self.action_space.n

    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return

    def step(self, action):
        self.num_steps += 1
        
        #move n steps from current position
        steps = self.actions[action]
        self.current_position = self.calculate_position(self.current_position, steps)
        self.absolute_position = self.calculate_position(self.current_position,self.position_bias)

        self.kept_rel_images.append(self.current_position)
        self.kept_abs_images.append(self.absolute_position) 

        #carve in new position (absolute)
        self.spc.carve(self.absolute_position) 

        #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
        h = np.histogram(self.spc.sc.values(), bins=3)[0]

        #calculate increment of detected spaces since last carving
        delta = h[0] - self.last_vspaces_count
        self.last_vspaces_count = h[0]

        reward = delta / 100.0

        if self.num_steps >= 9:
            self.done = True
           
        self.total_reward += reward

        self.current_state = ( self.spc.sc.values() , self.current_position )

        return self.current_state, reward, self.done, {}

 
    def minMaxNorm(self,old, oldmin, oldmax , newmin , newmax):
        return ( (old-oldmin)*(newmax-newmin)/(oldmax-oldmin) ) + newmin

   
  
    def calculate_position(self,init_state,steps):
        n_pos = init_state + steps
        if n_pos>(self.n_positions-1):
            n_pos -= self.n_positions
        elif n_pos<0:
            n_pos += self.n_positions
        return n_pos


