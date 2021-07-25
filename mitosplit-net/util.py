import numpy as np
import h5py
import pickle
from glob import glob
from tqdm import tqdm

import tensorflow as tf

#General
def get_filename(path, keyword, extension=None):
    if extension is None:
        return [full_path.split('\\')[-1] for full_path in glob(path+keyword+'*')]
    return [full_path.split('\\')[-1].replace('.'+extension, '') for full_path in glob(path+keyword+'*') if extension in full_path]

def save_pkl(data, path, name):
    filename = path+name
    print('\nSaving '+filename)
    pickle.dump(data, open(filename, 'wb'))
    print('Done.')
    
def load_pkl(path, name):
    try:
        filename = path+name
        print('\nLoading '+filename)
        return pickle.load(open(filename, 'rb'))      
    except:
        filename = [path+title for title in name]
        data = []
        for fname in filename:
            print('\nLoading '+fname)
            data += [pickle.load(open(fname, 'rb'))]
        return data      
    
def save_h5(data, path, name):
  filename = path+name+'.h5'
  print('\nSaving '+filename)
  hf = h5py.File(filename, 'a')
  hf= hf.create_dataset(name, data=data)
  print('Done.')
  
def load_h5(path, name):
    try:
        filename = path+name+'.h5'
        print('\nLoading '+filename)
        hf = h5py.File(filename, 'r').get(name)
        print('Converting to array')
        return np.array(hf)
    except:
        filename = [path+fname+'.h5' for fname in name]
        data = []
        for full_dir, fname in zip(filename, name):
            print('\nLoading '+full_dir)
            hf = h5py.File(full_dir, 'r').get(fname)
            data += [np.array(hf)]
        return np.array(data)
 

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
