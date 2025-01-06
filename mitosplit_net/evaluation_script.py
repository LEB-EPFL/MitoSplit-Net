#!/usr/bin/env python
# coding: utf-8

# # Evaluation of models on test data

# In[293]:


from IPython import get_ipython
ipython = get_ipython()
if ipython:
    ipython.magic("reload_ext autoreload")
    ipython.magic("autoreload 2")
prefix = '3_'
import util
import plotting
import evaluation

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=18)

from tqdm import tqdm
import tensorflow as tf


# In[294]:


#Define GPU device where the code will run on
gpu = tf.config.list_physical_devices('GPU')[0]
print(gpu)
tf.config.experimental.set_memory_growth(gpu, True)
gpu = tf.device('GPU:0/')


# ## Data and models directories

# In[295]:


base_dir = '/mnt/LEB/Scientific_projects/deep_events_WS/data/single_channel_fluo/MitoSplit-Net/'
data_path = base_dir+'Data/' 
model_path = base_dir+'Models/' 


# ## No preprocessing, different batch sizes

# ### Only mito channel as input

# In[296]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
print('Inputs'+':', input_data.shape)

#Outputs
output_data = util.load_h5(data_path, 'Proc')
print('Outputs:', output_data.shape)

labels = util.load_h5(data_path, 'labels')
print('Labels:', labels.shape)


# In[ ]:


from pathlib import Path
folder_name = util.get_filename(model_path, 'ref_f8')
folder_name = [folder for folder in folder_name if folder.split('/')[-1][:3] == 'ref']
idx_sort = np.argsort([int(model_name.split('_b')[-1]) for model_name in folder_name])
folder_name = [folder_name[i] for i in idx_sort]
folder_name = [str(Path(folder).parts[-1]) for folder in folder_name]
nb_models = len(folder_name)

model = util.load_model(model_path, [prefix + 'model']*nb_models, folder_name, as_type=dict)
history = util.load_pkl(model_path, ['history']*nb_models, folder_name, as_type=dict)
frames_test = util.load_pkl(model_path, ['frames_test']*nb_models, folder_name, as_type=dict)


# In[298]:


input_test, output_test, pred_output_test = {}, {}, {}
labels_test = {}

model_pbar = tqdm(model.keys())
for model_name in model_pbar:
  model_pbar.set_description("Processing %s" %model_name)
  input_test[model_name] = input_data[frames_test[model_name]]
  output_test[model_name] = output_data[frames_test[model_name]]
  labels_test[model_name] = labels[frames_test[model_name]]

  pred_output_test[model_name] = evaluation.predict(input_test[model_name], model[model_name])

del output_data, labels, input_data


# #### Threshold optimization

# In[ ]:


pred_threshold = np.linspace(0.05, 1, 10, endpoint=True)
f1_score = {}
optimal_pred_threshold = {}
pred_labels_test = {}

for model_name in model:
  print("Processing %s" %model_name)
  f1_score[model_name] = evaluation.get_f1_curve(labels_test[model_name], pred_output_test[model_name], pred_threshold)
  
  optimal_pred_threshold[model_name] = evaluation.get_optimal_threshold(pred_threshold, f1_score[model_name])
  
  pred_labels_test[model_name] = evaluation.label(pred_output_test[model_name], threshold=optimal_pred_threshold[model_name])
  print('\n')
  
util.save_pkl(pred_threshold, model_path, 'ref_pred_threshold')
util.save_pkl(f1_score, model_path, 'ref_f1_score')
util.save_pkl(optimal_pred_threshold, model_path, 'ref_optimal_pred_threshold')
print("\033[1m\033[31mref written\033[0m")


# In[300]:


batch_size = [int(model_name.split('_b')[-1]) for model_name in folder_name]

num_models = len(batch_size)
cbar_ticks = np.arange(num_models)

norm_bounds = -1, num_models-1
norm = Normalize(*norm_bounds)
cmap = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
colors = cmap.get_cmap()(norm(cbar_ticks))

fig, ax = plt.subplots(figsize=(7, 7))
for model_name, b, c in zip(model, batch_size, colors):
  ax.plot(pred_threshold, f1_score[model_name], 'o-', color=c, label=b)

ax.set(xlabel='Normalized event score threshold', ylabel='F1-score', ylim=(0, 1))
ax.legend(title='Batch size', title_fontsize=20)
plt.show()


# In[301]:


title = 'ref_examples'
filename = base_dir+'Figures/'+title+'.pdf'
print(filename)

title_size = 36

fig, axes = plt.subplots(2, nb_models, figsize=(4*nb_models, 8))
fig.suptitle('Batch size', size=title_size)
for model_name, b, i in zip(folder_name, [8, 16, 32, 256], range(nb_models)):
  frame = np.random.choice(np.where(np.any(np.any(output_test[model_name]>0, axis=-1), axis=-1))[0])
  mask = pred_output_test[model_name][frame]>optimal_pred_threshold[model_name]
  plotting.plot_merge(input_test[model_name][frame], output_test[model_name][frame], title='', ax=axes[0, i])
  plotting.plot_merge(input_test[model_name][frame], pred_output_test[model_name][frame]*mask, title='', ax=axes[1, i])
  axes[0, i].set_title(b, size=title_size)
  
fig.tight_layout(pad=0)
fig.subplots_adjust(wspace=0, hspace=0.02, top=0.85)
#plt.show()
fig.savefig(filename)


# In[ ]:


for model_name in folder_name:
  mask = pred_output_test[model_name]>optimal_pred_threshold[model_name]
  plotting.plot_outputs(input_test[model_name], output_test[model_name], 
                        pred_output_test[model_name]*mask, frames_test[model_name], 
                        nb_examples=5, title=model_name, cmap=['gray', 'inferno'])
  plt.show()
  print('\n')


# #### Labels binary overlap

# In[303]:


fissionStats = {}
for model_name in model:
  print('\nModel name:', model_name)
  fissionStats[model_name] = evaluation.fissionStatsStack(labels_test[model_name], pred_labels_test[model_name])
  
fissionStats = pd.DataFrame(fissionStats, index=['TP', 'FP', 'FN', 'TP_px', 'FP_px', 'FN_px']).T
fissionStats['precision'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FP'])
fissionStats['TPR'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FN'])
fissionStats['FDR'] = fissionStats['FP']/(fissionStats['TP']+fissionStats['FP'])

fissionStats['precision_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])
fissionStats['TPR_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FN_px'])
fissionStats['FDR_px'] = fissionStats['FP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])

fissionStats = fissionStats.T


# In[304]:


title = 'ref_fissionStats'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig = plt.figure(figsize=(7*2, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

axes = []
axes += [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])


plotting.plot_metrics_comparison(fissionStats.loc[['precision', 'TPR', 'FDR']], xscale=2, color=colors, ax=axes[0], ylim=[0, 1], legend=False)  
plotting.plot_metrics_comparison(fissionStats.loc[['precision_px', 'TPR_px', 'FDR_px']], xscale=2, color=colors, ax=axes[1], ylim=[0, 1], legend=False)
axes[1].set_yticklabels([])

for ax, title in zip(axes, ['Object level', 'Pixel level']):
  ax.set_title(title, size=22)
  ax.tick_params(axis='x')
  ax.tick_params(axis='y')

cbar = fig.colorbar(cmap, cax=cax, ticks=cbar_ticks)
cbar.set_ticklabels(batch_size)
cbar.set_label('Batch size (a.u.)', labelpad=15)


plt.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.05, top=0.84, right=0.98)

#plt.show()
fig.savefig(filename)


# #### Detection match

# In[305]:


det_match = [evaluation.detection_match(output_test[model_name], pred_output_test[model_name]) for model_name in folder_name]


# In[306]:


facecolor = (0.13, 0.13, 0.13, 1)
title = 'ref_det_match_new'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(det_match, 'o-')
ax.set_xticks(range(len(det_match)))
ax.set_xticklabels([8, 16, 32, 256])
ax.set_xlabel('Batch size')
ax.set_ylabel('Detection match')

ax.tick_params(axis='x')
ax.tick_params(axis='y')

plt.tight_layout(pad=0)
#plt.show()
fig.savefig(filename)


# ### Mito + Drp1 channels as inputs

# In[307]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
input_data = np.stack((input_data, util.load_h5(data_path, 'Drp1')), axis=-1)
print('Inputs'+':', input_data.shape)

#Outputs
output_data = util.load_h5(data_path, 'Proc')
print('Outputs:', output_data.shape)

labels = util.load_h5(data_path, 'labels')
print('Labels:', labels.shape)


# In[308]:


folder_name


# In[ ]:


folder_name = util.get_filename(model_path, 'multich_ref_f8')
idx_sort = np.argsort([int(model_name.split('_b')[-1]) for model_name in folder_name])

folder_name = [folder_name[i] for i in idx_sort]
folder_name = [str(Path(folder).parts[-1]) for folder in folder_name]
# folder_name = [folder for folder in folder_name if folder.split('/')[-1][:3] == 'mul']
nb_models = len(folder_name)

model = util.load_model(model_path, [prefix + 'model']*nb_models, folder_name, as_type=dict)
history = util.load_pkl(model_path, ['history']*nb_models, folder_name, as_type=dict)
frames_test = util.load_pkl(model_path, ['frames_test']*nb_models, folder_name, as_type=dict)


# In[310]:


input_test, output_test, pred_output_test = {}, {}, {}
labels_test = {}

model_pbar = tqdm(model.keys())
for model_name in model_pbar:
  model_pbar.set_description("Processing %s" %model_name)
  input_test[model_name] = input_data[frames_test[model_name]]
  output_test[model_name] = output_data[frames_test[model_name]]
  labels_test[model_name] = labels[frames_test[model_name]]

  pred_output_test[model_name] = evaluation.predict(input_test[model_name], model[model_name])

del output_data, labels, input_data


# #### Threshold optimization

# In[311]:


pred_threshold = np.linspace(0.05, 1, 10, endpoint=True)
f1_score = {}
optimal_pred_threshold = {}
pred_labels_test = {}

for model_name in model:
  print("Processing %s" %model_name)
  f1_score[model_name] = evaluation.get_f1_curve(labels_test[model_name], pred_output_test[model_name], pred_threshold)
  
  optimal_pred_threshold[model_name] = evaluation.get_optimal_threshold(pred_threshold, f1_score[model_name])
  
  pred_labels_test[model_name] = evaluation.label(pred_output_test[model_name], threshold=optimal_pred_threshold[model_name])
  print('\n')
  
util.save_pkl(f1_score, model_path, 'multich_ref_f1_score')
util.save_pkl(optimal_pred_threshold, model_path, 'multich_ref_optimal_pred_threshold')


# In[312]:


batch_size = [int(model_name.split('_b')[-1]) for model_name in folder_name]

num_models = len(batch_size)
cbar_ticks = np.arange(num_models)

norm_bounds = -1, num_models-1
norm = Normalize(*norm_bounds)
cmap = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
colors = cmap.get_cmap()(norm(cbar_ticks))

fig, ax = plt.subplots(figsize=(7, 7))
for model_name, b, c in zip(model, batch_size, colors):
  ax.plot(pred_threshold, f1_score[model_name], 'o-', color=c, label=b)

ax.set(xlabel='Normalized event score threshold', ylabel='F1-score', ylim=(0, 1))
ax.legend(title='Batch size', title_fontsize=20)
plt.show()


# In[313]:


plt.imshow(input_test[model_name][frame][:, :, 0])


# In[314]:


title = 'multich_ref_examples'
filename = base_dir+'Figures/'+title+'.pdf'
print(filename)

title_size = 36

fig, axes = plt.subplots(2, nb_models, figsize=(4*nb_models, 8))
fig.suptitle('Batch size', size=title_size)
for model_name, b, i in zip(folder_name, [8, 16, 32, 256], range(nb_models)):
  frame = np.random.choice(np.where(np.any(np.any(output_test[model_name]>0, axis=-1), axis=-1))[0])
  mask = pred_output_test[model_name][frame]>optimal_pred_threshold[model_name]
  plotting.plot_merge(input_test[model_name][frame][:, :, 0], output_test[model_name][frame], title='', ax=axes[0, i])
  plotting.plot_merge(input_test[model_name][frame][:, :, 0], pred_output_test[model_name][frame]*mask, title='', ax=axes[1, i])
  axes[0, i].set_title(b, size=title_size)
  
fig.tight_layout(pad=0)
fig.subplots_adjust(wspace=0, hspace=0.02, top=0.85)
#plt.show()
fig.savefig(filename)


# In[ ]:


for model_name in folder_name:
  mask = pred_output_test[model_name]>optimal_pred_threshold[model_name]
  plotting.plot_outputs(input_test[model_name][:, :, :, 0], output_test[model_name], 
                        pred_output_test[model_name]*mask, frames_test[model_name], 
                        nb_examples=5, title=model_name, cmap=['gray', 'inferno'])
  plt.show()
  print('\n')


# #### Labels binary overlap

# In[316]:


fissionStats = {}
for model_name in model:
  print('\nModel name:', model_name)
  fissionStats[model_name] = evaluation.fissionStatsStack(labels_test[model_name], pred_labels_test[model_name])
  
fissionStats = pd.DataFrame(fissionStats, index=['TP', 'FP', 'FN', 'TP_px', 'FP_px', 'FN_px']).T
fissionStats['precision'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FP'])
fissionStats['TPR'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FN'])
fissionStats['FDR'] = fissionStats['FP']/(fissionStats['TP']+fissionStats['FP'])

fissionStats['precision_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])
fissionStats['TPR_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FN_px'])
fissionStats['FDR_px'] = fissionStats['FP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])

fissionStats = fissionStats.T


# In[317]:


title = 'multich_ref_fissionStats'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig = plt.figure(figsize=(7*2, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

axes = []
axes += [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])


plotting.plot_metrics_comparison(fissionStats.loc[['precision', 'TPR', 'FDR']], xscale=2, color=colors, ax=axes[0], ylim=[0, 1], legend=False)  
plotting.plot_metrics_comparison(fissionStats.loc[['precision_px', 'TPR_px', 'FDR_px']], xscale=2, color=colors, ax=axes[1], ylim=[0, 1], legend=False)
axes[1].set_yticklabels([])

for ax, title in zip(axes, ['Object level', 'Pixel level']):
  ax.set_title(title, size=22)
  ax.tick_params(axis='x')
  ax.tick_params(axis='y')

cbar = fig.colorbar(cmap, cax=cax, ticks=cbar_ticks)
cbar.set_ticklabels(batch_size)
cbar.set_label('Batch size (a.u.)', labelpad=15)


plt.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.05, top=0.84, right=0.98)

#plt.show()
fig.savefig(filename)


# #### Detection match

# In[318]:


det_match = [evaluation.detection_match(output_test[model_name], pred_output_test[model_name]) for model_name in folder_name]


# In[319]:


facecolor = (0.13, 0.13, 0.13, 1)
title = 'multich_ref_det_match'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(det_match, 'o-')
ax.set_xticks(range(len(det_match)))
ax.set_xticklabels([8, 16, 32, 256])
ax.set_xlabel('Batch size')
ax.set_ylabel('Detection match')

ax.tick_params(axis='x')
ax.tick_params(axis='y')

plt.tight_layout(pad=0)
#plt.show()
fig.savefig(filename)


# ## Mito & WatProc, different spot sizes

# ### Only mito channel as input

# In[320]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
print('Inputs'+':', input_data.shape)

dilation_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
dilation_threshold = util.load_pkl(data_path, 'max_intensity_threshold')
num_sigmas = dilation_sigma.shape[0]


# In[321]:


#Outputs
folder_name = util.get_filename(model_path, 'wp_f8')
folder_name = [model_name for model_name in folder_name if np.all([tag not in model_name for tag in ['aug', 'temp', 'multich']])]
folder_name = [str(Path(folder).parts[-1]) for folder in folder_name]
num_models = len(folder_name)

model = util.load_model(model_path, [prefix + 'model']*num_models, folder_name, as_type=dict)
history = util.load_pkl(model_path, ['history']*num_models, folder_name, as_type=dict)
frames_test = util.load_pkl(model_path, ['frames_test']*num_models, folder_name, as_type=dict)


# In[322]:


input_test, output_test, pred_output_test = {}, {}, {}
labels_test = {}

for model_name, s, t in zip(folder_name, dilation_sigma, dilation_threshold):
  print('\nModel: %s'%model_name)
  
  #Inputs
  input_test[model_name] = input_data[frames_test[model_name]]
  
  metadata = 's%.1f_t%.i'%(s, t)
  #Outputs
  output_data = util.load_h5(data_path, 'WatProc_'+metadata)
  print('Outputs:', output_data.shape)
  output_test[model_name] = output_data[frames_test[model_name]]
  del output_data
  
  #Labels
  labels = util.load_h5(data_path, 'proc_labels_'+metadata)
  print('Labels:', labels.shape)
  labels_test[model_name] = labels[frames_test[model_name]]
  del labels
  
  pred_output_test[model_name] = evaluation.predict(input_test[model_name], model[model_name])

del input_data


# #### Threshold optimization

# In[323]:


pred_threshold = np.linspace(0.05, 1, 10, endpoint=True)
f1_score = {}
optimal_pred_threshold = {}
pred_labels_test = {}

for model_name in model:
  print("Processing %s" %model_name)
  f1_score[model_name] = evaluation.get_f1_curve(labels_test[model_name], pred_output_test[model_name], pred_threshold)
  optimal_pred_threshold[model_name] = evaluation.get_optimal_threshold(pred_threshold, f1_score[model_name])
  pred_labels_test[model_name] = evaluation.label(pred_output_test[model_name], threshold=optimal_pred_threshold[model_name])
  print('\n')
  
util.save_pkl(pred_threshold, model_path, 'pred_threshold')
util.save_pkl(f1_score, model_path, 'wp_f1_score')
util.save_pkl(optimal_pred_threshold, model_path, 'wp_optimal_pred_threshold')
print("\033[1m\033[31mwp written\033[0m")


# In[324]:


cbar_ticks = np.arange(num_models)

norm_bounds = -1, num_models-1
norm = Normalize(*norm_bounds)
cmap = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
colors = cmap.get_cmap()(norm(cbar_ticks))

fig, ax = plt.subplots(figsize=(10, 10))
for model_name, thr, c in zip(model, dilation_threshold, colors):
  ax.plot(pred_threshold, f1_score[model_name], 'o-', color=c, label=thr)

ax.set(xlabel='Normalized event score threshold', ylabel='F1-score', ylim=(0, 1))
ax.legend(title='Event score threshold\nafter dilation', title_fontsize=20, ncol=2)
plt.show()


# In[ ]:


for model_name in folder_name:
  mask = pred_output_test[model_name]>optimal_pred_threshold[model_name]
  plotting.plot_outputs(input_test[model_name], output_test[model_name], 
                        pred_output_test[model_name]*mask, frames_test[model_name], 
                        nb_examples=5, title=model_name, cmap=['gray', 'inferno'])
  plt.show()
  print('\n')


# #### Labels binary overlap

# In[326]:


fissionStats = {}
for model_name in labels_test:
  print('\nModel name:', model_name)
  fissionStats[model_name] = evaluation.fissionStatsStack(labels_test[model_name], pred_labels_test[model_name])


# In[327]:


fissionStats = pd.DataFrame(fissionStats, index=['TP', 'FP', 'FN', 'TP_px', 'FP_px', 'FN_px']).T
fissionStats['precision'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FP'])
fissionStats['TPR'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FN'])
fissionStats['FDR'] = fissionStats['FP']/(fissionStats['TP']+fissionStats['FP'])

fissionStats['precision_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])
fissionStats['TPR_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FN_px'])
fissionStats['FDR_px'] = fissionStats['FP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])

fissionStats = fissionStats.T


# In[328]:


title = 'wp_fissionStats'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig = plt.figure(figsize=(7*2, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

axes = []
axes += [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])

plotting.plot_metrics_comparison(fissionStats.loc[['precision', 'TPR', 'FDR']], xscale=2, color=colors, ax=axes[0], ylim=[0, 1], legend=False)  
plotting.plot_metrics_comparison(fissionStats.loc[['precision_px', 'TPR_px', 'FDR_px']], xscale=2, color=colors, ax=axes[1], ylim=[0, 1], legend=False)
axes[1].set_yticklabels([])

for ax, title in zip(axes, ['Object level', 'Pixel level']):
  ax.set_title(title, size=22)
  ax.tick_params(axis='x')
  ax.tick_params(axis='y')

cbar = fig.colorbar(cmap, cax=cax, ticks=cbar_ticks)
cbar.set_ticklabels(dilation_threshold)
cbar.set_label('Event score threshold\nafter dilation (a.u.)', labelpad=15)


plt.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.05, top=0.84, right=0.98)

#plt.show()
fig.savefig(filename)


# #### Detection match

# In[329]:


det_match = [evaluation.detection_match(output_test[model_name], pred_output_test[model_name]) for model_name in folder_name]


# In[330]:


title = 'wp_det_match'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(dilation_threshold, det_match, 'o-')
ax.set(xlabel='Event score threshold (a.u.)', ylabel='Detection match')
#plt.show()
fig.savefig(filename)


# ### Mito + Drp1 channels as inputs

# In[331]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
input_data = np.stack((input_data, util.load_h5(data_path, 'Drp1')), axis=-1)
print('Inputs'+':', input_data.shape)

dilation_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
dilation_threshold = util.load_pkl(data_path, 'max_intensity_threshold')
num_sigmas = dilation_sigma.shape[0]


# In[ ]:


#Outputs
folder_name = util.get_filename(model_path, 'multich_wp_f8')
folder_name = [model_name for model_name in folder_name if np.all([tag not in model_name for tag in ['aug', 'temp']])]
folder_name = [str(Path(folder).parts[-1]) for folder in folder_name]
num_models = len(folder_name)

model = util.load_model(model_path, [prefix + 'model']*num_models, folder_name, as_type=dict)
history = util.load_pkl(model_path, ['history']*num_models, folder_name, as_type=dict)
frames_test = util.load_pkl(model_path, ['frames_test']*num_models, folder_name, as_type=dict)


# In[333]:


input_test, output_test, pred_output_test = {}, {}, {}
labels_test = {}

for model_name, s, t in zip(folder_name, dilation_sigma, dilation_threshold):
  print('\nModel: %s'%model_name)
  
  #Inputs
  input_test[model_name] = input_data[frames_test[model_name]]
  
  metadata = 's%.1f_t%.i'%(s, t)
  #Outputs
  output_data = util.load_h5(data_path, 'WatProc_'+metadata)
  print('Outputs:', output_data.shape)
  output_test[model_name] = output_data[frames_test[model_name]]
  del output_data
  
  #Labels
  labels = util.load_h5(data_path, 'proc_labels_'+metadata)
  print('Labels:', labels.shape)
  labels_test[model_name] = labels[frames_test[model_name]]
  del labels
  
  pred_output_test[model_name] = evaluation.predict(input_test[model_name], model[model_name])

del input_data


# #### Threshold optimization

# In[334]:


pred_threshold = np.linspace(0.05, 1, 10, endpoint=True)
f1_score = {}
optimal_pred_threshold = {}
pred_labels_test = {}

for model_name in model:
  print("Processing %s" %model_name)
  f1_score[model_name] = evaluation.get_f1_curve(labels_test[model_name], pred_output_test[model_name], pred_threshold)
  optimal_pred_threshold[model_name] = evaluation.get_optimal_threshold(pred_threshold, f1_score[model_name])
  pred_labels_test[model_name] = evaluation.label(pred_output_test[model_name], threshold=optimal_pred_threshold[model_name])
  print('\n')
  
util.save_pkl(f1_score, model_path, 'multich_wp_f1_score')
util.save_pkl(optimal_pred_threshold, model_path, 'multich_wp_optimal_pred_threshold')


# In[335]:


cbar_ticks = np.arange(num_models)

norm_bounds = -1, num_models-1
norm = Normalize(*norm_bounds)
cmap = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
colors = cmap.get_cmap()(norm(cbar_ticks))

fig, ax = plt.subplots(figsize=(10, 10))
for model_name, thr, c in zip(model, dilation_threshold, colors):
  ax.plot(pred_threshold, f1_score[model_name], 'o-', color=c, label=thr)

ax.set(xlabel='Normalized event score threshold', ylabel='F1-score', ylim=(0, 1))
ax.legend(title='Event score threshold\nafter dilation', title_fontsize=20, ncol=2)
plt.show()


# In[ ]:


for model_name in folder_name:
  mask = pred_output_test[model_name]>optimal_pred_threshold[model_name]
  plotting.plot_outputs(input_test[model_name][:, :, :, 0], output_test[model_name], 
                        pred_output_test[model_name]*mask, frames_test[model_name], 
                        nb_examples=5, title=model_name, cmap=['gray', 'inferno'])
  plt.show()
  print('\n')


# #### Labels binary overlap

# In[337]:


fissionStats = {}
for model_name in labels_test:
  print('\nModel name:', model_name)
  fissionStats[model_name] = evaluation.fissionStatsStack(labels_test[model_name], pred_labels_test[model_name])


# In[338]:


fissionStats = pd.DataFrame(fissionStats, index=['TP', 'FP', 'FN', 'TP_px', 'FP_px', 'FN_px']).T
fissionStats['precision'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FP'])
fissionStats['TPR'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FN'])
fissionStats['FDR'] = fissionStats['FP']/(fissionStats['TP']+fissionStats['FP'])

fissionStats['precision_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])
fissionStats['TPR_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FN_px'])
fissionStats['FDR_px'] = fissionStats['FP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])

fissionStats = fissionStats.T


# In[339]:


title = 'multich_wp_fissionStats'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig = plt.figure(figsize=(7*2, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

axes = []
axes += [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])

plotting.plot_metrics_comparison(fissionStats.loc[['precision', 'TPR', 'FDR']], xscale=2, color=colors, ax=axes[0], ylim=[0, 1], legend=False)  
plotting.plot_metrics_comparison(fissionStats.loc[['precision_px', 'TPR_px', 'FDR_px']], xscale=2, color=colors, ax=axes[1], ylim=[0, 1], legend=False)
axes[1].set_yticklabels([])

for ax, title in zip(axes, ['Object level', 'Pixel level']):
  ax.set_title(title, size=22)
  ax.tick_params(axis='x')
  ax.tick_params(axis='y')

cbar = fig.colorbar(cmap, cax=cax, ticks=cbar_ticks)
cbar.set_ticklabels(dilation_threshold)
cbar.set_label('Event score threshold\nafter dilation (a.u.)', labelpad=15)


plt.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.05, top=0.84, right=0.98)

#plt.show()
fig.savefig(filename)


# #### Detection match

# In[340]:


det_match = [evaluation.detection_match(output_test[model_name], pred_output_test[model_name]) for model_name in folder_name]


# In[341]:


title = 'multich_wp_det_match'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(dilation_threshold, det_match, 'o-')
ax.set(xlabel='Event score threshold (a.u.)', ylabel='Detection match')
#plt.show()
fig.savefig(filename)


# ## Spatiotemporal filtered Mito & WatProc

# ### Only mito channel as input

# In[ ]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
print('Inputs'+':', input_data.shape)

dilation_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
dilation_threshold = util.load_pkl(data_path, 'max_intensity_threshold')
num_sigmas = dilation_sigma.shape[0]


# In[ ]:


#Outputs
folder_name = util.get_filename(model_path, 'spatemp_wp')
folder_name = [str(Path(folder).parts[-1]) for folder in folder_name]
folder_name = [folder for folder in folder_name if 'multich' not in folder]
folder_name = [folder for folder in folder_name if 'optimal' not in folder]
folder_name = [folder for folder in folder_name if 'f1_score' not in folder]
num_models = len(folder_name)

model = util.load_model(model_path, [prefix + 'model']*num_models, folder_name, as_type=dict)
history = util.load_pkl(model_path, ['history']*num_models, folder_name, as_type=dict)
frames_test = util.load_pkl(model_path, ['frames_test']*num_models, folder_name, as_type=dict)


# In[344]:


input_test, output_test, pred_output_test = {}, {}, {}
labels_test = {}

for model_name in folder_name:
  print('\nModel: %s'%model_name)
  metadata = model_name.split('b16_')[-1]
  output_data_name = 'spatemp_WatProc_'+metadata
  labels_name = 'spatemp_proc_labels_'+metadata
  
  #Outputs
  output_data = util.load_h5(data_path, output_data_name)
  print('Outputs:', output_data.shape)
  output_test[model_name] = output_data[frames_test[model_name]]
  del output_data

  labels = util.load_h5(data_path, labels_name)
  print('Labels:', labels.shape)
  labels_test[model_name] = labels[frames_test[model_name]]
  del labels
  
  input_test[model_name] = input_data[frames_test[model_name]]
  pred_output_test[model_name] = evaluation.predict(input_test[model_name], model[model_name])
  
del input_data


# #### Threshold optimization

# In[345]:


pred_threshold = util.load_pkl(model_path, 'pred_threshold')
f1_score = {}
optimal_pred_threshold = {}
pred_labels_test = {}

for model_name in model:
  print("Processing %s" %model_name)
  f1_score[model_name] = evaluation.get_f1_curve(labels_test[model_name], pred_output_test[model_name], pred_threshold)
  optimal_pred_threshold[model_name] = evaluation.get_optimal_threshold(pred_threshold, f1_score[model_name])
  pred_labels_test[model_name] = evaluation.label(pred_output_test[model_name], threshold=optimal_pred_threshold[model_name])
  print('\n')
  
util.save_pkl(f1_score, model_path, 'spatemp_wp_f1_score')
util.save_pkl(optimal_pred_threshold, model_path, 'spatemp_wp_optimal_pred_threshold')
print("\033[1m\033[31mspatemp written\033[0m")


# In[346]:


cbar_ticks = np.arange(num_models)

norm_bounds = -1, num_models-1
norm = Normalize(*norm_bounds)
cmap = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
colors = cmap.get_cmap()(norm(cbar_ticks))

fig, ax = plt.subplots(figsize=(10, 10))
for model_name, thr, c in zip(model, dilation_threshold, colors):
  ax.plot(pred_threshold, f1_score[model_name], 'o-', color=c, label=thr)

ax.set(xlabel='Normalized event score threshold', ylabel='F1-score', ylim=(0, 1))
ax.legend(title='Event score threshold\nafter dilation', title_fontsize=20, ncol=2)
plt.show()


# In[ ]:


for model_name in folder_name:
  mask = pred_output_test[model_name]>optimal_pred_threshold[model_name]
  plotting.plot_outputs(input_test[model_name], output_test[model_name], 
                        pred_output_test[model_name]*mask, frames_test[model_name], 
                        nb_examples=5, title=model_name, cmap=['gray', 'inferno'])
  plt.show()
  print('\n')


# #### Labels binary overlap

# In[348]:


fissionStats = {}
for model_name in labels_test:
  print('\nModel name:', model_name)
  fissionStats[model_name] = evaluation.fissionStatsStack(labels_test[model_name], pred_labels_test[model_name])


# In[349]:


fissionStats = pd.DataFrame(fissionStats, index=['TP', 'FP', 'FN', 'TP_px', 'FP_px', 'FN_px']).T
fissionStats['precision'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FP'])
fissionStats['TPR'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FN'])
fissionStats['FDR'] = fissionStats['FP']/(fissionStats['TP']+fissionStats['FP'])

fissionStats['precision_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])
fissionStats['TPR_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FN_px'])
fissionStats['FDR_px'] = fissionStats['FP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])

fissionStats = fissionStats.T


# In[350]:


gaussian_threshold = np.array([int(model_name.split('_gt')[-1]) for model_name in model])
title = 'spatemp_wp_fissionStats'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig = plt.figure(figsize=(7*2, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

axes = []
axes += [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])

plotting.plot_metrics_comparison(fissionStats.loc[['precision', 'TPR', 'FDR']], xscale=2, color=colors, ax=axes[0], ylim=[0, 1], legend=False)  
plotting.plot_metrics_comparison(fissionStats.loc[['precision_px', 'TPR_px', 'FDR_px']], xscale=2, color=colors, ax=axes[1], ylim=[0, 1], legend=False)
axes[1].set_yticklabels([])

for ax, title in zip(axes, ['Object level', 'Pixel level']):
  ax.set_title(title, size=22)
  ax.tick_params(axis='x')
  ax.tick_params(axis='y')

cbar = fig.colorbar(cmap, cax=cax, ticks=cbar_ticks)
cbar.set_ticklabels(gaussian_threshold)
cbar.set_label('Event score threshold (a.u.)', labelpad=15)


plt.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.05, top=0.84, right=0.98)

#plt.show()
fig.savefig(filename)


# #### Detection match

# In[351]:


det_match = [evaluation.detection_match(output_test[model_name], pred_output_test[model_name]) for model_name in folder_name]


# In[352]:


title = 'spatemp_wp_det_match'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(dilation_threshold, det_match, 'o-')
ax.set(xlabel='Event score threshold (a.u.)', ylabel='Detection match')
#plt.show()
fig.savefig(filename)


# ### Mito + Drp1 channels as inputs

# In[353]:


#Inputs
input_data = util.load_h5(data_path, 'Mito')
input_data = np.stack((input_data, util.load_h5(data_path, 'Drp1')), axis=-1)
print('Inputs'+':', input_data.shape)

dilation_sigma = util.load_pkl(data_path, 'max_optimal_sigma')
dilation_threshold = util.load_pkl(data_path, 'max_intensity_threshold')
num_sigmas = dilation_sigma.shape[0]


# In[354]:


#Outputs
folder_name = util.get_filename(model_path, 'multich_spatemp_wp_f8')
folder_name = [str(Path(folder).parts[-1]) for folder in folder_name]
num_models = len(folder_name)

model = util.load_model(model_path, [prefix + 'model']*num_models, folder_name, as_type=dict)
history = util.load_pkl(model_path, ['history']*num_models, folder_name, as_type=dict)
frames_test = util.load_pkl(model_path, ['frames_test']*num_models, folder_name, as_type=dict)


# In[355]:


input_test, output_test, pred_output_test = {}, {}, {}
labels_test = {}

for model_name in folder_name:
  print('\nModel: %s'%model_name)
  metadata = model_name.split('b16_')[-1]
  output_data_name = 'spatemp_WatProc_'+metadata
  labels_name = 'spatemp_proc_labels_'+metadata
  
  #Outputs
  output_data = util.load_h5(data_path, output_data_name)
  print('Outputs:', output_data.shape)
  output_test[model_name] = output_data[frames_test[model_name]]
  del output_data

  labels = util.load_h5(data_path, labels_name)
  print('Labels:', labels.shape)
  labels_test[model_name] = labels[frames_test[model_name]]
  del labels
  
  input_test[model_name] = input_data[frames_test[model_name]]
  pred_output_test[model_name] = evaluation.predict(input_test[model_name], model[model_name])
  
del input_data


# #### Threshold optimization

# In[356]:


pred_threshold = util.load_pkl(model_path, 'pred_threshold')
f1_score = {}
optimal_pred_threshold = {}
pred_labels_test = {}

for model_name in model:
  print("Processing %s" %model_name)
  f1_score[model_name] = evaluation.get_f1_curve(labels_test[model_name], pred_output_test[model_name], pred_threshold)
  optimal_pred_threshold[model_name] = evaluation.get_optimal_threshold(pred_threshold, f1_score[model_name])
  pred_labels_test[model_name] = evaluation.label(pred_output_test[model_name], threshold=optimal_pred_threshold[model_name])
  print('\n')
  
util.save_pkl(f1_score, model_path, 'multich_spatemp_f1_score')
util.save_pkl(optimal_pred_threshold, model_path, 'multich_spatemp_optimal_pred_threshold')


# In[357]:


cbar_ticks = np.arange(num_models)

norm_bounds = -1, num_models-1
norm = Normalize(*norm_bounds)
cmap = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
colors = cmap.get_cmap()(norm(cbar_ticks))

fig, ax = plt.subplots(figsize=(10, 10))
for model_name, thr, c in zip(model, dilation_threshold, colors):
  ax.plot(pred_threshold, f1_score[model_name], 'o-', color=c, label=thr)

ax.set(xlabel='Normalized event score threshold', ylabel='F1-score', ylim=(0, 1))
ax.legend(title='Event score threshold\nafter dilation', title_fontsize=20, ncol=2)
plt.show()


# In[ ]:


for model_name in folder_name:
  mask = pred_output_test[model_name]>optimal_pred_threshold[model_name]
  plotting.plot_outputs(input_test[model_name][:, :, :, 0], output_test[model_name], 
                        pred_output_test[model_name]*mask, frames_test[model_name], 
                        nb_examples=5, title=model_name, cmap=['gray', 'inferno'])
  plt.show()
  print('\n')


# #### Labels binary overlap

# In[359]:


fissionStats = {}
for model_name in labels_test:
  print('\nModel name:', model_name)
  fissionStats[model_name] = evaluation.fissionStatsStack(labels_test[model_name], pred_labels_test[model_name])


# In[360]:


fissionStats = pd.DataFrame(fissionStats, index=['TP', 'FP', 'FN', 'TP_px', 'FP_px', 'FN_px']).T
fissionStats['precision'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FP'])
fissionStats['TPR'] = fissionStats['TP']/(fissionStats['TP']+fissionStats['FN'])
fissionStats['FDR'] = fissionStats['FP']/(fissionStats['TP']+fissionStats['FP'])

fissionStats['precision_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])
fissionStats['TPR_px'] = fissionStats['TP_px']/(fissionStats['TP_px']+fissionStats['FN_px'])
fissionStats['FDR_px'] = fissionStats['FP_px']/(fissionStats['TP_px']+fissionStats['FP_px'])

fissionStats = fissionStats.T


# In[361]:


gaussian_threshold = np.array([int(model_name.split('_gt')[-1]) for model_name in model])
title = 'multich_spatemp_fissionStats'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig = plt.figure(figsize=(7*2, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

axes = []
axes += [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])

plotting.plot_metrics_comparison(fissionStats.loc[['precision', 'TPR', 'FDR']], xscale=2, color=colors, ax=axes[0], ylim=[0, 1], legend=False)  
plotting.plot_metrics_comparison(fissionStats.loc[['precision_px', 'TPR_px', 'FDR_px']], xscale=2, color=colors, ax=axes[1], ylim=[0, 1], legend=False)
axes[1].set_yticklabels([])

for ax, title in zip(axes, ['Object level', 'Pixel level']):
  ax.set_title(title, size=22)
  ax.tick_params(axis='x')
  ax.tick_params(axis='y')

cbar = fig.colorbar(cmap, cax=cax, ticks=cbar_ticks)
cbar.set_ticklabels(gaussian_threshold)
cbar.set_label('Event score threshold (a.u.)', labelpad=15)


plt.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.05, top=0.84, right=0.98)

#plt.show()
fig.savefig(filename)


# #### Detection match

# In[362]:


det_match = [evaluation.detection_match(output_test[model_name], pred_output_test[model_name]) for model_name in folder_name]


# In[363]:


title = 'multich_spatemp_wp_det_match'
filename = base_dir+'Figures/'+title+'.png'
print(filename)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(dilation_threshold, det_match, 'o-')
ax.set(xlabel='Event score threshold (a.u.)', ylabel='Detection match')
#plt.show()
fig.savefig(filename)


# In[ ]:


print("\033[1m\033[31mDONE DONE DONE DONE\033[0m")

