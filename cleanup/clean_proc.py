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

# %%
red_proc = orig_proc[keep]
red_mito = orig_mito[keep]
viewer.add_image(red_mito)
viewer.add_image(red_proc, colormap="viridis", blending="additive")

# %%
corr_proc = copy.deepcopy(red_proc)
corr_proc_bin = corr_proc > 0
corr_proc_bin = corr_proc_bin.astype(np.uint8)

print("Deleting points")
del_points = viewer.layers["Delete"].data
for point in tqdm(del_points):
    point = point.astype(np.uint32)
    value_below = red_proc[point[0], point[1], point[2]]
    if value_below > 0:
        corr_proc_bin[point[0]] = flood_fill(corr_proc_bin[point[0]], (point[1], point[2]), 2)
        corr_proc[corr_proc_bin == 2] = 0

print("Adding points")
add_points = viewer.layers["Add"].data
sigma = 5.
for point in tqdm(add_points):
    empty_frame = np.zeros_like(corr_proc[int(point[0])])
    gauss = get_gaussian((point[1], point[2]), (sigma, sigma),
                         (corr_proc.shape[1], corr_proc.shape[2]))
    gauss = gauss.numpy()
    gauss = gauss/np.max(gauss)
    corr_proc[int(point[0])] = corr_proc[int(point[0])] + gauss.astype(np.float32)
# %%
viewer.add_image(corr_proc, colormap="viridis", blending="additive")


# %%
import tensorflow_probability as tfp
import tensorflow as tf
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
