import numpy as np
import h5py
import pickle
from glob import glob
import os
from tqdm import tqdm

import tensorflow as tf

#General
def get_filename(path, keyword, extension=None):
    if extension is None:
        return [full_path.split('\\')[-1] for full_path in glob(path+'*') if keyword in full_path]
    return [full_path.split('\\')[-1].replace('.'+extension, '') for full_path in glob(path+'*') if (keyword in full_path)&(extension in full_path)]

def save_pkl(data, path, name, folder_name=None):
    try:
        if folder_name is not None:
            filename = path+folder_name+name
        else:
            filename = path+name
        print('\nSaving '+filename)
        pickle.dump(data, open(filename, 'wb'))
        print('Done.')
    except:
        if folder_name is not None and len(folder_name)!=len(name):
            raise ValueError("'name' and 'folder_name' lenghts don't match.")
        elif folder_name is not None and len(folder_name)==len(name):
            for file, subfolder, title in zip(data, folder_name, name):
                if not os.path.exists(path+subfolder):
                    os.makedirs(path+subfolder)
                filename = path+subfolder+'/'+title
                print('\nSaving '+filename)
                if type(data)==dict:
                    pickle.dump(data[file], open(filename, 'wb'))
                else:
                    pickle.dump(file, open(filename, 'wb'))
            print('Done.')
        else:
            for file, title in zip(data, name):
                filename = path+title
                print('\nSaving '+filename)
                pickle.dump(file, open(filename, 'wb'))
            print('Done.')
        
            
    
def load_pkl(path, name, folder_name=None, as_type=None):
    try:
        try:
            filename = path+folder_name+'/'+name
        except:
            filename = path+name
        print('\nLoading '+filename)
        return pickle.load(open(filename, 'rb'))      
    except:
        filename = [path+title for title in name]
        data = []
        
        if folder_name is not None and len(folder_name)!=len(name):
            raise ValueError("'name' and 'folder_name' lenghts don't match.")
        elif folder_name is not None and len(folder_name)==len(name):
            for subfolder, title in zip(folder_name, name):
                filename = path+subfolder+'/'+title
                print('\nLoading '+filename)
                data += [pickle.load(open(filename, 'rb'))]
            print('Done.')
        else:
            for title in name:
                filename = path+title
                print('\nLoading '+filename)
                data += [pickle.load(open(filename, 'rb'))]
            print('Done.')
        if as_type==np.ndarray:
            return np.array(data)
        elif as_type==dict:
            try:
                return dict(zip(folder_name, data))
            except:
                return dict(zip(name, data))
        else:
            return data
    
def save_h5(data, path, name):
    filename = path+name+'.h5'
    print('\nSaving '+filename)
    hf = h5py.File(filename, 'a')
    hf= hf.create_dataset(name, data=data, compression="gzip", compression_opts=9)
    print('Done.')
  
def load_h5(path, name, indices=None):
    try:
        filename = path+name+'.h5'
        print('\nLoading '+filename)
        hf = h5py.File(filename, 'r').get(name)
        print('Converting to array')
        if indices is None:
            return np.array(hf)
        else:
            return hf[indices]
    except:
        data = []
        if indices is None:
            for title in name:
                filename = path+title+'.h5'
                print('\nLoading '+filename)
                hf = h5py.File(filename, 'r').get(title)
                data += [np.array(hf)]
            return np.array(data)
        else:
            for title in name:
                filename = path+title+'.h5'
                print('\nLoading '+filename)
                hf = h5py.File(filename, 'r').get(title)
                data += [hf[indices]]
            return np.array(data)

def save_model(model, path, name, folder_name=None):
    try:
        return model.save(path+name+'.h5')
    except:
        if folder_name is not None and len(folder_name)!=len(name):
            raise ValueError("'name' and 'folder_name' lenghts don't match.")
        elif folder_name is not None and len(folder_name)==len(name):
            for model_name, subfolder, title in zip(model, folder_name, name):
                if not os.path.exists(path+subfolder):
                    os.makedirs(path+subfolder)
                filename = path+subfolder+'/'+title+'.h5'
                print('\nSaving '+filename)
                model[model_name].save(filename)
            print('Done.')
        else:
            for model_name, title in zip(data, name):
                filename = path+title+'.h5'
                print('\nSaving '+filename)
                model[model_name].save(filename)
            print('Done.')
    return model

def load_model(path, name, folder_name=None, as_type=None, all_models=False):
    try:
        try:
            filename = path+folder_name+'/'+name
        except:
            filename = path+name
        print('\nLoading '+filename)
        return tf.keras.models.load_model(filename+'.h5')
    except:
        model = []

        if all_models:
            all_models_dir = glob(model_path+'*/model.h5')

            for model_dir in tqdm(all_models_dir, total=len(all_models_dir)):
                model_name = model_dir.split('\\')[-2]
                print('\nLoading %s'%model_name)
                model[model_name] = tf.keras.models.load_model(model_name)
            return model

        if folder_name is not None and len(folder_name)!=len(name):
            raise ValueError("'name' and 'folder_name' lenghts don't match.")
        elif folder_name is not None and len(folder_name)==len(name):
            for subfolder, title in zip(folder_name, name):
                filename = path+subfolder+'/'+title+'.h5'
                print('\nLoading '+filename)
                model += [tf.keras.models.load_model(filename)]
            print('Done.')
        else:
            for title in name:
                filename = path+title+'.h5'
                print('\nLoading '+filename)
                model += [tf.keras.models.load_model(filename)]
            print('Done.')

        if as_type==np.ndarray:
            return np.array(model)
        elif as_type==dict:
            try:
                return dict(zip(folder_name, model))
            except:
                return dict(zip(name, model))
        else:
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
