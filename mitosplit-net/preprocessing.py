import numpy as np
from skimage import filters, segmentation, feature, measure
import scipy.ndimage as ndi

def distance_watershed(img):
  """Segmentation of events of interest based on Distance Transform Watershed 
  of the Hessian probability map of divisions"""
  distance = ndi.distance_transform_edt(img)
  #Division sites as makers
  coords = feature.peak_local_max(distance, labels=img>0)
  mask = np.zeros(distance.shape, dtype=bool)
  mask[tuple(coords.T)] = True
  markers = ndi.label(mask)[0]
  #Watershed
  return segmentation.watershed(-distance, markers, mask=img)

def prepareProc(img, threshold=0, vmax=255):
  """Smoothed probability map of divisions"""
  labels = distance_watershed(img)  
  labels = segmentation.clear_border(labels)
  img2 = np.zeros_like(img)
  if len(np.unique(labels))>1: #Pixels above the threshold and not in contact with the border
    # Gaussian dilation to increase minimal spot size
    rp = measure.regionprops(labels)
    centroids = np.array([region.centroid for region in rp]).round().astype(int)
    mask = (centroids[:, 0], centroids[:, 1])
    img2[mask] = img[mask]
    img2 = filters.gaussian(img2, 2)
    #Normalization
    img2 = img2*vmax/img2.max()
  return img2

def prepareStack(stack, **kwargs):
  """Apply prepareProc to all of the images in stack"""
  if stack.ndim == 2:
    return prepareProc(stack, **kwargs)
  procStack = np.zeros_like(stack)
  for i, img in tqdm(enumerate(stack)):
    procStack[i] = prepareProc(img, **kwargs)
  return procStack