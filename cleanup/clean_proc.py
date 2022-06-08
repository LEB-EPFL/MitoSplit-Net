#%%
from mitosplit_net.util import load_h5
import h5py
import napari
import numpy as np
import copy
from skimage.segmentation import flood, flood_fill
from scipy import signal
from tqdm import tqdm

# %% Load the Proc and clean up the data augmentation
orig_proc_file = "W:/Watchdog/MitoSplit-Net/Data/"
name = list(h5py.File(orig_proc_file + "Output.h5", 'r').keys())
orig_proc = h5py.File(orig_proc_file + "Output.h5", 'r').get(name[-1])
orig_mito = load_h5(orig_proc_file, "Mito")

# %%
viewer = napari.Viewer()

#%%
# viewer.add_image(orig_mito)
# viewer.add_image(orig_proc, colormap="viridis", blending="additive")

#%% Construct subset that is the data without the rotation
keep = np.empty(0)
for rotation in range(orig_proc.shape[0]//1000):
    keep = np.append(keep, int(rotation * 1000) + np.arange(0,100))
keep = keep.astype(np.uint32)

# %% Put this data into napari
red_proc = orig_proc[keep]
red_mito = orig_mito[keep]
viewer.add_image(red_mito)
viewer.add_image(red_proc, colormap="viridis", blending="additive")

# %% Use the "Delete" and "Add" layers to modify the data
corr_proc = copy.deepcopy(red_proc)
corr_proc_bin = corr_proc > 0
corr_proc_bin = corr_proc_bin.astype(np.uint8)

# Delete points in the "Delete" layer
del_points = viewer.layers["Delete"].data
for point in tqdm(del_points, desc="Delete"):
    point = point.astype(np.uint32)
    value_below = red_proc[point[0], point[1], point[2]]
    if value_below > 0:
        corr_proc_bin[point[0]] = flood_fill(corr_proc_bin[point[0]], (point[1], point[2]), 2)
        corr_proc[corr_proc_bin == 2] = 0

# Add points from the "Add" layer
add_points = viewer.layers["Add"].data
sigma = 5.
for point in tqdm(add_points, desc="Add"):
    empty_frame = np.zeros_like(corr_proc[int(point[0])])
    gauss = get_gaussian((point[1], point[2]), (sigma, sigma),
                         (corr_proc.shape[1], corr_proc.shape[2]))
    gauss = gauss.numpy()
    gauss = gauss/np.max(gauss)
    gauss[gauss < 0.1] = 0
    gauss = gauss/np.max(gauss)
    corr_proc[int(point[0])] = corr_proc[int(point[0])] + gauss.astype(np.float32)
# %% Update Proc in napari
viewer.add_image(corr_proc, colormap="viridis", blending="additive")

#%% Save this data
from mitosplit_net.util import save_h5
save_h5(corr_proc, orig_proc_file, "Proc_manual")

#%%[markdown]
# Clean up this data

#%% Imports for overall clean up
import mitosplit_net.preprocessing as preprocessing
from skimage import measure, transform

#%% Clean up overall data with binary GT
sigma = 3.9
clean_proc = np.zeros_like(corr_proc)
for frame in tqdm(range(corr_proc.shape[0]), desc="Cleaning up"):
    labels = preprocessing.distance_watershed(corr_proc[frame], sigma=sigma/10)

    rp = measure.regionprops_table(labels, intensity_image=corr_proc[frame], properties=['weighted_centroid'])
    centroids = np.array([rp['weighted_centroid-0'], rp['weighted_centroid-1']])
    for point in range(centroids.shape[1]):
        gauss = get_gaussian((centroids[0, point], centroids[1, point]), (sigma, sigma),
                            (clean_proc.shape[1], clean_proc.shape[2]))
        gauss = gauss.numpy()
        gauss = gauss/np.max(gauss)
        gauss[gauss < 0.1] = 0
        gauss = gauss/np.max(gauss)
        clean_proc[frame] = clean_proc[frame] + gauss.astype(np.float32)

#%% Put the rotation back into the Proc in order to use it with the original rotated data
final_proc = np.zeros((clean_proc.shape[0]*10, clean_proc.shape[1], clean_proc.shape[2]))
for submovie in tqdm(range(clean_proc.shape[0]//100)):
    for rot in range(10):
        for frame in range(100):
            start_rot_set = rot*100+submovie*1000
            final_proc[start_rot_set + frame] = transform.rotate(clean_proc[submovie*100 + frame],-rot*10,
                                                                       resize=False, preserve_range=True)

#%% Look in napari
viewer.add_image(orig_mito)
viewer.add_image(final_proc, colormap="viridis", blending="additive")

#%% Save that final Proc
save_h5(corr_proc, orig_proc_file, "Proc_manual_full")


# %%
import tensorflow_probability as tfp
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
#%%
def get_gaussian(mu, sigma, size):
    mu = ((mu[1]+0.5-0.5*size[1])/(size[1]*0.5), (mu[0]+0.5-0.5*size[0])/(size[0]*0.5))
    sigma = (sigma[0]/size[0], sigma[1]/size[1])
    mvn = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    x,y = tf.cast(tf.linspace(-1,1,size[0]),tf.float32), tf.cast(tf.linspace(-1,1,size[1]), tf.float32)
    # meshgrid as a list of [x,y] coordinates
    coords = tf.reshape(tf.stack(tf.meshgrid(x,y),axis=-1),(-1,2)).numpy()
    gauss = mvn.prob(coords)
    return tf.reshape(gauss, size)
# %%
