import numpy as np
from skimage import filters, segmentation, feature, measure
import scipy.ndimage as ndi
from tqdm import tqdm

def distance_watershed(img, sigma=1):
  """Segmentation of events of interest based on Distance Transform Watershed of the Hessian probability map of divisions"""
  img_smooth = filters.gaussian(img) 
  distance = ndi.distance_transform_edt(img_smooth)
  #Division sites as makers
  coords = feature.peak_local_max(distance, labels=img_smooth>0)
  mask = np.zeros(distance.shape, dtype=bool)
  mask[tuple(coords.T)] = True
  markers = ndi.label(mask)[0]
  #Watershed
  return segmentation.watershed(-distance, markers, mask=img)

def prepareProc(img, sigma=1, vmax=255):
  """Smoothed probability map of divisions"""
  labels = distance_watershed(img)  
  labels = segmentation.clear_border(labels) #Remove labels in contact with the border
  img_proc = np.zeros_like(img)
  if len(np.unique(labels))>1: 
    # Gaussian dilation to increase minimal spot size
    fission_props = measure.regionprops_table(labels, properties=['centroid', 'equivalent_diameter'])    
    fission_props['centroid-0'] = fission_props['centroid-0'].round().astype(int)
    fission_props['centroid-1'] = fission_props['centroid-1'].round().astype(int)
    mask = (fission_props['centroid-0'], fission_props['centroid-1'])
    img_proc[mask] = img[mask]
    img_proc = filters.gaussian(img_proc, sigma)
    #Normalization
    img_proc = img_proc*vmax/img.max()
    return img_proc, fission_props
  return img_proc, {}

def prepareStack(stack, **kwargs):
  """Apply prepareProc to all of the images in stack"""
  if stack.ndim == 2:
    return prepareProc(stack, **kwargs)
  procStack = np.zeros_like(stack)
  fission_props = np.zeros(stack.shape[0], dtype=object)
  for i, img in tqdm(enumerate(stack), total=stack.shape[0]):
    procStack[i], fission_props[i] = prepareProc(img, **kwargs)
  return procStack, fission_props