#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('reload_ext', 'tensorboard')
import numpy as np
import json
import tensorflow as tf
import os

import time
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import gym

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
  except RuntimeError as e:
    print(e)
    
    
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_logical_devices('GPU')

import tf_agents
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import py_environment, parallel_py_environment, batched_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function, element_wise_squared_loss
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.policies import policy_saver
import logging

import tensorflow.keras as keras

tf.compat.v1.enable_v2_behavior()
import time
import json
import datetime
import copy
import shutil

#import imp
from scan_gym import envs
#imp.reload(envs)
import csv

seed=42
tf.random.set_seed(seed)
np.random.seed(seed)


# In[ ]:


current_path = os.getcwd()
params_file = os.path.join(current_path, 'params.json') 
pm=json.load(open(params_file))
run_label = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_log_path = os.path.join(current_path, 'generated_data/') 


# In[ ]:


'''pt = '028'
data_paths = ['/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/ramdisk/'+pt,
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/ramdisk/'+pt,
             '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/ramdisk/'+pt,
            '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/ramdisk/'+pt]'''

'''data_paths = ['/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/001',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/003',
             '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/025',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/039',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/041',
              
             '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/006',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/009',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/011',
             '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/024',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/022',
              
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/000',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/020',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/028',
             '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/036',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/048',
              
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/201',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/202',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/203',
             '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/204',
              '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/200',
             ]'''

data_paths = ['','','']

envs = [suite_gym.load('ScannerEnv-v1',gym_kwargs={'dataset_path':dp,'gt_mode':True,'rotation_steps':0,'init_pos':-1}) 
        for dp in data_paths]

tf_env = tf_py_environment.TFPyEnvironment( batched_py_environment.BatchedPyEnvironment(envs,multithreading=True) )#, isolation=True  )

env_name = 'ScannerEnv-v0'
if env_name in gym.envs.registry.env_specs:
    print('s')
    del gym.envs.registry.env_specs[env_name]
# In[ ]:


tf_env.observation_spec()


# In[ ]:


tf_env.action_spec()


# In[ ]:


'''def volume_layers():
    input_vol = keras.layers.Input(shape=(66,68,152))
    preprocessing = keras.layers.Reshape((66,68,152,1))(input_vol)
    preprocessing = keras.layers.Cropping3D(cropping=((0,2), (0,4), (0,24)))(preprocessing) #output none,64,64,128
    preprocessing = keras.layers.Lambda(lambda x: (tf.cast(x,np.float32)+1.) / 2.)(preprocessing) #normalize 0-1
    
    
    x = keras.layers.Conv3D(filters=32, kernel_size=3,strides=1, padding="same", activation="relu")(preprocessing)
    x = keras.layers.Conv3D(filters=16, kernel_size=1,strides=1, padding="same", activation="relu")(x)
    x = keras.layers.Conv3D(filters=64, kernel_size=3,strides=2,padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)
   
    #x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    
    #x = keras.layers.Dense(32)(x)
                                        
    model = keras.models.Model(inputs=input_vol,outputs=x)
    model.summary()
    return model'''

def volume_layers():
    input_vol = keras.layers.Input(shape=(66,68,152))
    preprocessing = keras.layers.Reshape((66,68,152,1))(input_vol)
    preprocessing = keras.layers.Cropping3D(cropping=((0,2), (0,4), (0,24)))(preprocessing) #output none,64,64,128
    preprocessing = keras.layers.Lambda(lambda x: (tf.cast(x,np.float32)+1.) / 2.)(preprocessing) #normalize 0-1
    stride = 2
    
    x = keras.layers.Conv3D(filters=16, kernel_size=5,strides=stride, padding="same", activation="relu")(preprocessing)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Conv3D(filters=32, kernel_size=3,strides=stride, padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=64, kernel_size=3,strides=stride,padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
    #x = keras.layers.BatchNormalization()(x)
    
    #x = keras.layers.Conv3D(filters=64, kernel_size=3,strides=stride,padding="same", activation="relu")(x)
    #x = keras.layers.MaxPool3D(pool_size=2)(x)
   
    #x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    
    #x = keras.layers.Dense(32)(x)
                                        
    model = keras.models.Model(inputs=input_vol,outputs=x)
    model.summary()
    return model
    
    
#scale range 0 to 1
oldmin = tf_env.observation_spec()[1].minimum
oldmax = tf_env.observation_spec()[1].maximum
print(oldmin,oldmax)
    
def input_vect_layers():
    input_ = keras.layers.Input(shape=(1,))
    preprocessing = keras.layers.Lambda(lambda x: ((x-oldmin)*(1.- 0.)/(oldmax-oldmin)) + 0. )(input_)
    #x = keras.layers.Dense(32)(preprocessing)
    return keras.models.Model(inputs=input_,outputs=preprocessing)


# In[ ]:


#network
#preprocessing_layers=volume_layers()
preprocessing_layers=(volume_layers(),input_vect_layers())
preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
dense_l = pm['model']['fc_layer_params']
if len(dense_l) == 1:
    fc_layer_params = (dense_l[0],)
else:
    fc_layer_params = dense_l


categorical_q_net = categorical_q_network.CategoricalQNetwork(
tf_env.observation_spec(),
tf_env.action_spec(),
preprocessing_layers=preprocessing_layers,
preprocessing_combiner=preprocessing_combiner,
fc_layer_params=fc_layer_params,
num_atoms=pm['categorical_dqn']['n_atoms'])


# In[ ]:


#agent
train_step = tf.Variable(0)
#optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
#            epsilon=0.00001, centered=True)
'''lr_decay =  keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.005, # initial ε
            decay_steps = pm['agent']['decay_steps'], 
            end_learning_rate=0.0005)
optimizer = keras.optimizers.Adam(learning_rate=lambda: lr_decay(train_step))'''


optimizer = keras.optimizers.Adam(learning_rate=pm['model']['learning_rate'])

epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0, # initial ε
            decay_steps = pm['agent']['decay_steps'], 
            end_learning_rate=0.02) # final ε

agent = categorical_dqn_agent.CategoricalDqnAgent(tf_env.time_step_spec(),
                tf_env.action_spec(),
                categorical_q_network=categorical_q_net,
                optimizer=optimizer,
                min_q_value=pm['categorical_dqn']['min_q_value'],
                max_q_value=pm['categorical_dqn']['max_q_value'],
                target_update_period=pm['agent']['target_update_period'],
                td_errors_loss_fn=keras.losses.Huber(reduction="none"),#element_wise_squared_loss,
                gamma=pm['agent']['gamma'], # discount factor
                train_step_counter=train_step,
                n_step_update =  pm['categorical_dqn']['n_step_update'],                                
                epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()


# In[ ]:


#Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec= agent.collect_data_spec,
    batch_size= tf_env.batch_size,
    max_length=pm['rbuffer']['max_length'])


# In[ ]:


#observer
#observer is just a function (or a callable object) that takes a trajectory argument,
#add_method() method (bound to the replay_buffer object) can be used as observer
replay_buffer_observer = replay_buffer.add_batch


# In[ ]:


#observer for training metrics
training_metrics = [
tf_metrics.NumberOfEpisodes(),
tf_metrics.AverageEpisodeLengthMetric(batch_size=len(envs)),
tf_metrics.EnvironmentSteps(),
tf_metrics.AverageReturnMetric(batch_size=len(envs)),
tf_metrics.MaxReturnMetric(batch_size=len(envs)),
tf_metrics.MinReturnMetric(batch_size=len(envs)),   
]


# In[ ]:


#custom observer
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


# In[ ]:


#Collect Driver
update_period = pm['collect_driver']['num_steps'] # train the model every x steps
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + training_metrics,
    num_steps=update_period) # collect x steps for each training iteration

#+ training_metrics,


# In[ ]:


# random policy driver to start filling the buffer
random_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                        tf_env.action_spec())

ns = pm['rnd_policy']['num_steps']
init_driver = DynamicStepDriver(
            tf_env,
            random_collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=ns)
#, ShowProgress(ns)],
final_time_step, final_policy_state = init_driver.run()


# In[ ]:


#use buffer as tf API dataset ()
dataset = replay_buffer.as_dataset(
        sample_batch_size=pm['rbuffer']['sample_batch_size'],
        num_steps=pm['categorical_dqn']['n_step_update'] + 1,
        num_parallel_calls=3).prefetch(3)


# In[ ]:


#convert main functions to tensorflow functions to speed up training
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


# In[ ]:


#@tf.function
def train_agent(n_iterations):
    #reset metrics
    for m in training_metrics:
        m.reset()
    time_step = None
    policy_state = ()#agent.collect_policy.get_initial_state(tf_env.batch_size)
    agent.train_step_counter.assign(0)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f} epsilon{}".format(iteration, train_loss.loss.numpy(),epsilon_fn(train_step)), end="")
        if iteration % 100 == 0:
            with train_summary_writer.as_default():
                #plot metrics
                for train_metric in training_metrics:
                    train_metric.tf_summaries(train_step=tf.cast(agent.train_step_counter,tf.int64), step_metrics=training_metrics[:])
                train_summary_writer.flush()
                
        if iteration % 100 == 0:
            with train_summary_writer.as_default():
                #plot train loss          
                tf.summary.scalar('train_loss', train_loss.loss.numpy(), step=tf.cast(agent.train_step_counter,tf.int64))
                #plot NN weights
                for layer in  categorical_q_net.layers:
                    for weight in layer.weights:
                        tf.summary.histogram(weight.name,weight,step=tf.cast(agent.train_step_counter,tf.int64))
                train_summary_writer.flush()


# In[ ]:


train_dir = os.path.join(data_log_path,"logs/",run_label)   
train_summary_writer = tf.summary.create_file_writer(
            train_dir, flush_millis=10000)
#train_summary_writer.set_as_default()


# In[ ]:


# Launch TensorBoard with objects in the log directory
# This should launch tensorboard in your browser, but you may not see your metadata.
#%tensorboard --logdir=logs --reload_interval=15


# In[ ]:


#tf.summary.scalar('avgreturn', training_metrics[3].result().numpy(), step=tf.cast(agent.train_step_counter,tf.int64))


# In[ ]:


train_agent(pm['misc']['n_iterations'])


# In[ ]:


#save model, parameters and code state
policy_dir = os.path.join(data_log_path,"policies", run_label)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)

with open(os.path.join(data_log_path,"parameters", run_label+'.json'), 'w') as json_file:
  json.dump(pm, json_file)

src = os.path.join(current_path,"categorical_dqn_tf_agents.ipynb")
dst = os.path.join(data_log_path,"train_code", run_label+'.ipynb')
shutil.copyfile(src, dst)

src = os.path.join(current_path,"scan_gym/scan_gym/envs/ScannerEnv/scanner_env.py")
dst = os.path.join(data_log_path,"environment_code", run_label+'.py')
shutil.copyfile(src, dst)


# In[ ]:


def run_episode(env,tf_env,policy):
    state = tf_env.reset()
    time_steps = 40
    for i in range(1,time_steps):
        action_step = policy.action(state)
        state = tf_env.step(action_step.action)
        #print(i,':',action_step.action.numpy(),state.observation[1].numpy()[0],state.step_type[0].numpy())  #,state.step_type.numpy()  )#,env.action2move(action_step.action,state.observation[0][-1].numpy()))#action,dir
        last_position = env.current_position
        if state.is_last():
            break
    return env.h

def test_envs(envs,tf_envs,policy):
    stats = []
    similarity_ratios = []
    for  model in range(len(envs)):
        hs = []
        sr = []
        for i in range(180):
            h = run_episode(envs[model], tf_envs[model],policy)
            sr.append(envs[model].last_gt_ratio)
            hs.append(h)
        stats.append(np.mean(hs,axis=0).astype('int'))
        similarity_ratios.append(np.mean(sr))
        print(model)
    return stats,similarity_ratios

def save_tests(path,stats,similarity_ratios,models):
    f = open(path,'w')
    writer = csv.writer(f)
    writer.writerow(["model","empty", "undefined", "solid", "similarity"])
    for i in range(len(stats)):
        writer.writerow([models[i]] + stats[i].tolist() + [similarity_ratios[i]])
    stats_mean = np.mean(stats,axis=0)
    similarity_mean = np.mean(similarity_ratios)
    writer.writerow([9999] + stats_mean.astype('int').tolist() + [similarity_mean])    #(np.insert(total_mean,0,999).astype('int'))
    f.close()


# In[ ]:



models = [0,20,28,204,205]#[204,202]#,79,88,146,198,199]
envs = [suite_gym.load('ScannerEnv-v1',
        gym_kwargs={'dataset_path':os.path.join(current_path,'arabidopsis_im_bigger',str(x).zfill(3)),'init_pos_inc_rst':True, 'gt_mode':True}) for x in models]

tf_envs = [tf_py_environment.TFPyEnvironment( env ) for env in envs]


dest_dir = os.path.join(data_log_path,'tests')
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

stats,similarities = test_envs(envs,tf_envs,agent.policy)    
test_file = os.path.join(dest_dir,run_label + '.csv')
save_tests(test_file,stats,similarities,models)
print(stats, similarities)


# In[ ]:


replay_buffer.num_frames()


# In[ ]:


dp = '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/204'
env = suite_gym.load('ScannerEnv-v1',gym_kwargs={'dataset_path':dp,'rotation_steps':0,'init_pos':-1})
test_env = tf_py_environment.TFPyEnvironment( env )


# In[ ]:


state = test_env.reset()
#print(state.observation[0].numpy())
cont = 0
time_steps = 25
#obs = deque(maxlen=time_steps)
#print('0:',action_step.action.numpy(),state.observation[1].numpy()[0],state.step_type[0].numpy())

for i in range(1,time_steps):
    action_step = agent.policy.action(state)#agent.policy.action(state)
    state = test_env.step(action_step.action)
    print(i,':',action_step.action.numpy(),state.observation[1].numpy()[0],state.step_type[0].numpy())  #,state.step_type.numpy()  )#,env.action2move(action_step.action,state.observation[0][-1].numpy()))#action,dir
    #print(i,':',action_step.action.numpy() )#,state.observation.numpy()[0],state.step_type[0].numpy())
    last_position = env.current_position
    if state.is_last():
        print('salio',cont)
        break
        
print(env.kept_images)
print(sorted(env.kept_images))
print(env.total_reward)
print('bias',env.position_bias)
print(env.h)
print(env.last_gt_ratio)

#policy_dir = os.path.join(current_path, 'policy')
#policy = tf.compat.v2.saved_model.load(policy_dir)
# In[ ]:





# In[ ]:


#batched_py_environment.BatchedPyEnvironment(envs=[suite.load(...) for _ in range(n)])


# In[ ]:





# In[ ]:


#tf_env.observation_spec()[1].maximum


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#random_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
#                        tf_env.action_spec())


# In[ ]:


#random_collect_policy.collect_data_spec


# In[ ]:


#q_net.layers[0].weights[0].shape


# In[ ]:


#random_collect_policy.time_step_spec


# In[ ]:


'''ns = pm['rnd_policy']['num_steps']
init_driver = DynamicStepDriver(
            tf_env,
            random_collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=ns)'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#import tf_agents.utils
#tf_agents.utils.nest_utils.get_outer_shape(state,  tf_env.time_step_spec())


# In[ ]:


#agent.training_data_spec


# In[ ]:


#tf_env.observation_spec()[1].minimum


# In[ ]:




