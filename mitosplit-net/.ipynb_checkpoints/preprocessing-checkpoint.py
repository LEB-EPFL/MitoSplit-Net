import numpy as np
from skimage import filters, segmentation, feature, measure, morphology
import scipy.ndimage as ndi
from tqdm import tqdm

def distance_watershed(img, coords=None, sigma=0.1):
    """Segmentation of events of interest based on Distance Transform Watershed of the Hessian probability map of divisions"""
    img_smooth = filters.gaussian(img, sigma) #Smoothing so local maxima are well-defined
    distance = ndi.distance_transform_edt(img_smooth)
    
    if coords is None:
        #Division sites as makers
        coords = feature.peak_local_max(distance, labels=img_smooth>0)
        coords = tuple(coords.T)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[coords] = True
    markers = ndi.label(mask)[0]
    #Watershed
    return segmentation.watershed(-distance, markers, mask=img)

def segmentFissions(img, fission_props, sigma=0):
    """Segmentation of processed fissions using original fission sites as markers for distace Watershed."""
    if len(fission_props)>0:
        coords = (fission_props['centroid-0'], fission_props['centroid-1'])
        return distance_watershed(img, coords, sigma)       
    return np.zeros_like(img)

def segmentFissionsStack(stack, fission_props, **kwargs):
    """Iterates segmentFissions."""
    if stack.ndim==2:
        return segmentFissions(stack, fission_props, **kwargs)
    
    nb_img = stack.shape[0]
    labels = np.zeros_like(stack)
    for i in tqdm(range(nb_img), total=nb_img):
        labels[i] = segmentFissions(stack[i], fission_props[i], **kwargs)
    return labels
        

def analyzeFissions(labels):
    """Find fission sites and measure their diameter"""
    fission_props = {}
    if np.any(labels!=0): 
        fission_props = measure.regionprops_table(labels, properties=['centroid', 'equivalent_diameter'])    
        fission_props['centroid-0'] = fission_props['centroid-0'].round().astype(int)
        fission_props['centroid-1'] = fission_props['centroid-1'].round().astype(int)
    return fission_props
    
def analyzeFissionsStack(labels):
    """Iterates analyzeFissions."""
    if labels.ndim==2:
        return analyzeFissions(labels, **kwargs)
    
    nb_img = labels.shape[0]
    fission_props = [{}]*nb_img
    for i in tqdm(range(nb_img), total=nb_img):
        fission_props[i] = analyzeFissions(labels[i])
    return fission_props

def prepareProc(img, coords=None, sigma=1, vmax=255, min_size=9):
  """Smoothed probability map of divisions"""
  mask = segmentation.clear_border(img>0) #Remove objects in contact with the border
  img_proc = img*mask 
  labels = distance_watershed(img_proc, coords, 0.1*sigma)
  labels = morphology.remove_small_objects(labels, min_size) #Remove too small objects
  img_proc = np.zeros_like(img)
  if np.any(labels!=0): 
    # Gaussian dilation to increase minimal spot size
    fission_props = measure.regionprops_table(labels, properties=['centroid', 'equivalent_diameter'])    
    fission_props['centroid-0'] = fission_props['centroid-0'].round().astype(int)
    fission_props['centroid-1'] = fission_props['centroid-1'].round().astype(int)
    mask = (fission_props['centroid-0'], fission_props['centroid-1'])
    img_proc[mask] = img[mask]
    img_proc = filters.gaussian(img_proc, sigma)
    #Normalization
    img_proc = img_proc*vmax/img_proc.max()
    img_proc = img_proc*(img_proc>filters.threshold_otsu(img))
    return img_proc, fission_props
  return img_proc, {}

def prepareStack(stack, **kwargs):
  """Apply prepareProc to all of the images in stack"""
  if stack.ndim==2:
    return prepareProc(stack, **kwargs)
  
  stack_proc = np.zeros_like(stack)
  fission_props = [0]*stack.shape[0]
  for i, img in tqdm(enumerate(stack), total=stack.shape[0]):
    stack_proc[i], fission_props[i] = prepareProc(img, **kwargs)
  return stack_proc, fission_props
