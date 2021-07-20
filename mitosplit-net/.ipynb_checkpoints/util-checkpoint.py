import numpy as np
import h5py
import pickle
from glob import glob
from tqdm import tqdm

import tensorflow as tf


#General
def save_pkl(data, path, name):
    filename = path+name
    print('Saving '+filename)
    pickle.dump(data, open(filename, 'wb'))
    print('Done.')
    
def load_pkl(path, name):
    filename = path+name
    print('Loading '+filename)
    return pickle.load(open(filename, 'rb'))      
    
def save_h5(data, path, name):
  filename = path+name+'.h5'
  print('Saving '+filename)
  hf = h5py.File(filename, 'a')
  hf= hf.create_dataset(name, data=data)
  print('Done.')
  
def load_h5(path, name):
  filename = path+name+'.h5'
  print('\nLoading '+filename)
  hf = h5py.File(filename, 'r').get(name)
  print('Converting to array')
  return np.array(hf)

def load_model(model_path):
  try:
    return tf.keras.models.load_model(model_path+'.h5')
  except:
    all_models_dir = glob(model_path+'*/model.h5')
    model = {}
    for model_dir in tqdm(all_models_dir, total=len(all_models_dir)):
        model_id = model_dir.split('\\')[-2]
        model[model_id] = tf.keras.models.load_model(model_dir)
    return model        
    
def activity_percent(binary_output):
  """Percentage of active or inactive periods in binary_output"""
  N = len(binary_output)
  yes_counts = np.sum(binary_output)
  no_counts = N-yes_counts
  return 100*yes_counts/N, 100*no_counts/N

def norm_histogram(bins, rv):
  """Normalized histogram with binomial errors."""
  b = np.diff(bins)
  hist = np.histogram(rv, bins)[0]
  N = int(np.sum(hist))
  M = (b*hist).sum() #normalización
  hist = hist/M
  f_i = b*hist
  error = np.sqrt(N*f_i*(1-f_i))/M
  bincenters = (bins[1:] + bins[:-1])/2
  return bincenters, hist, error

def sequence_length(binary_signal):
  # make sure all runs of ones are well-bounded
  bounded = np.hstack(([0], binary_signal, [0]))
  # get 1 at run starts and -1 at run ends
  difs = np.diff(bounded)
  starts, = np.where(difs > 0)
  ends, = np.where(difs < 0)
  return ends - starts
