#!/usr/bin/env python
# coding: utf-8

# # Create and train MitoSplit-Net model

# ## Import required Python libraries
# 
# 

# In[1]:


import util
import plotting
import training

prefix = '3_'

import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=18)
from tqdm import tqdm

import tensorflow as tf


# In[2]:


#Define GPU device where the code will run on
gpu = tf.config.list_physical_devices('GPU')[0]
print(gpu)
tf.config.experimental.set_memory_growth(gpu, True)
gpu = tf.device('GPU:0/')


# ## Data and models directories

# In[3]:


base_dir = '/mnt/LEB/Scientific_projects/deep_events_WS/data/single_channel_fluo/MitoSplit-Net/'
print('base_dir:', base_dir)

data_path = base_dir+'Data/'
print('data_path:', data_path)

model_path = base_dir+'Models/'
print('model_path:', model_path)


# ## Create model, split dataset and train

# ### No preprocessing, different batch sizes

# #### Only mito channel as input

# In[4]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
print('Inputs'+':', input_data.shape)

#Outputs
output_data = util.load_h5(data_path, 'Proc')
print('Outputs:', output_data.shape)


# In[ ]:


with gpu:
  nb_filters = 8
  firstConvSize = 9
  batch_size = [8, 16, 32, 256]
  model, history, frames_test = {}, {}, {}
  
  for b in batch_size:
    model_name = 'ref_f%i_c%i_b%i'%(nb_filters, firstConvSize, b)
    print('Model:', model_name)
    model[model_name] = training.create_model(nb_filters, firstConvSize)
    history[model_name], frames_test[model_name] = training.train_model(model[model_name], input_data, output_data, batch_size=b) 


# In[ ]:


folder_name = list(model.keys())

util.save_model(model, model_path, [prefix + 'model']*len(model), folder_name)
util.save_pkl(history, model_path, [prefix + 'history']*len(model), folder_name)
util.save_pkl(frames_test, model_path, [prefix + 'frames_test']*len(model), folder_name)


# #### Mito + Drp1 channels as inputs

# In[7]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
input_data = np.stack((input_data, util.load_h5(data_path, 'Drp1')), axis=-1)
print('Inputs'+':', input_data.shape)

#Outputs
output_data = util.load_h5(data_path, 'Proc')
print('Outputs:', output_data.shape)


# In[ ]:


with gpu:
  nb_filters = 8
  firstConvSize = 9
  nb_input_channels = 2
  batch_size = [8, 16, 32, 256]
  model, history, frames_test = {}, {}, {}
  
  for b in batch_size:
    model_name = 'multich_ref_f%i_c%i_b%i'%(nb_filters, firstConvSize, b)
    print('Model:', model_name)
    model[model_name] = training.create_model(nb_filters, firstConvSize, nb_input_channels)
    history[model_name], frames_test[model_name] = training.train_model(model[model_name], input_data, output_data, batch_size=b) 


# In[ ]:


folder_name = list(model.keys())

util.save_model(model, model_path, [prefix + 'model']*len(model), folder_name)
util.save_pkl(history, model_path, [prefix + 'history']*len(model), folder_name)
util.save_pkl(frames_test, model_path, [prefix + 'frames_test']*len(model), folder_name)


# ### Mito & WatProc, different gaussian dilations

# #### Only mito channel as input

# In[10]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
print('Inputs'+':', input_data.shape)


# In[ ]:


with gpu:
  nb_filters = 8
  firstConvSize = 9
  batch_size = 16
  
  optimal_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
  threshold = util.load_pkl(data_path, 'max_intensity_threshold')
  model, history, frames_test = {}, {}, {}
  
  for s, t in zip(optimal_sigma, threshold):
    model_name = 'wp_f%i_c%i_b%i_s%.1f_t%.i'%(nb_filters, firstConvSize, batch_size, s, t)
    print('Model:', model_name)
    #Outputs
    output_data = util.load_h5(data_path, 'WatProc_s%.1f_t%.i'%(s, t))
    print('Outputs:', output_data.shape)
    model[model_name] = training.create_model(nb_filters, firstConvSize)
    
    history[model_name], frames_test[model_name] = training.train_model(model[model_name], input_data, output_data, batch_size=batch_size) 


# In[ ]:


folder_name = list(model.keys())

util.save_model(model, model_path, [prefix + 'model']*len(model), folder_name)
util.save_pkl(history, model_path, [prefix + 'history']*len(model), folder_name)
util.save_pkl(frames_test, model_path, [prefix + 'frames_test']*len(model), folder_name)


# In[13]:


del input_data, output_data


# #### Mito + Drp1 channels as inputs

# In[14]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
input_data = np.stack((input_data, util.load_h5(data_path, 'Drp1')), axis=-1)
print('Inputs'+':', input_data.shape)


# In[ ]:


with gpu:
  nb_filters = 8
  firstConvSize = 9
  nb_input_channels = 2
  batch_size = 16
  
  optimal_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
  threshold = util.load_pkl(data_path, 'max_intensity_threshold')
  model, history, frames_test = {}, {}, {}
  
  for s, t in zip(optimal_sigma, threshold):
    model_name = 'multich_wp_f%i_c%i_b%i_s%.1f_t%.i'%(nb_filters, firstConvSize, batch_size, s, t)
    print('Model:', model_name)
    #Outputs
    output_data = util.load_h5(data_path, 'WatProc_s%.1f_t%.i'%(s, t))
    print('Outputs:', output_data.shape)
    model[model_name] = training.create_model(nb_filters, firstConvSize, nb_input_channels)
    
    history[model_name], frames_test[model_name] = training.train_model(model[model_name], input_data, output_data, batch_size=batch_size) 


# In[ ]:


folder_name = list(model.keys())

util.save_model(model, model_path, [prefix + 'model']*len(model), folder_name)
util.save_pkl(history, model_path, [prefix + 'history']*len(model), folder_name)
util.save_pkl(frames_test, model_path, [prefix + 'frames_test']*len(model), folder_name)


# In[17]:


del input_data, output_data


# ## Spatiotemporal gaussian filter

# ### Only mito channel as input

# In[18]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
print('Inputs'+':', input_data.shape)


# In[ ]:


with gpu:
  nb_filters = 8
  firstConvSize = 9
  batch_size = 16
  
  dilation_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
  dilation_threshold = util.load_pkl(data_path, 'max_intensity_threshold')
  event_score_gaussian_threshold = util.load_pkl(data_path, 'event_score_gaussian_thresholds')
  time_window_size = 5
  
  model, history, frames_test = {}, {}, {}
  
  for sigma, threshold, gaussian_threshold in zip(dilation_sigma, dilation_threshold, event_score_gaussian_threshold):
    metadata = 's%.1f_t%i_w%i_gt%i'%(sigma, threshold, time_window_size, gaussian_threshold)
    model_name = 'spatemp_wp_f%i_c%i_b%i_'%(nb_filters, firstConvSize, batch_size) + metadata
    print('Model:', model_name)

    #Output
    output_data = util.load_h5(data_path, 'spatemp_WatProc_'+metadata)
    print('Output:', output_data.shape)
    model[model_name] = training.create_model(nb_filters, firstConvSize)

    history[model_name], frames_test[model_name] = training.train_model(model[model_name], input_data, output_data, batch_size=batch_size) 
    del output_data


# In[ ]:


folder_name = list(model.keys())

util.save_model(model, model_path, [prefix + 'model']*len(model), folder_name)
util.save_pkl(history, model_path, [prefix + 'history']*len(model), folder_name)
util.save_pkl(frames_test, model_path, [prefix + 'frames_test']*len(model), folder_name)


# ### Mito + Drp1 channels as inputs

# In[21]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
input_data = np.stack((input_data, util.load_h5(data_path, 'Drp1')), axis=-1)
print('Inputs'+':', input_data.shape)


# In[ ]:


with gpu:
  nb_filters = 8
  firstConvSize = 9
  nb_input_channels = 2
  batch_size = 16
  
  dilation_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
  dilation_threshold = util.load_pkl(data_path, 'max_intensity_threshold')
  event_score_gaussian_threshold = util.load_pkl(data_path, 'event_score_gaussian_thresholds')
  time_window_size = 5
  
  model, history, frames_test = {}, {}, {}
  
  for sigma, threshold, gaussian_threshold in zip(dilation_sigma, dilation_threshold, event_score_gaussian_threshold):
    metadata = 's%.1f_t%i_w%i_gt%i'%(sigma, threshold, time_window_size, gaussian_threshold)
    model_name = 'multich_spatemp_wp_f%i_c%i_b%i_'%(nb_filters, firstConvSize, batch_size) + metadata
    print('Model:', model_name)

    #Output
    output_data = util.load_h5(data_path, 'spatemp_WatProc_'+metadata)
    print('Output:', output_data.shape)
    model[model_name] = training.create_model(nb_filters, firstConvSize, nb_input_channels)

    history[model_name], frames_test[model_name] = training.train_model(model[model_name], input_data, output_data, batch_size=batch_size) 
    del output_data


# In[ ]:


folder_name = list(model.keys())

util.save_model(model, model_path, [prefix + 'model']*len(model), folder_name)
util.save_pkl(history, model_path, [prefix + 'history']*len(model), folder_name)
util.save_pkl(frames_test, model_path, [prefix + 'frames_test']*len(model), folder_name)

