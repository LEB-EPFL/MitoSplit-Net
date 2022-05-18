import numpy as np
from skimage import filters, segmentation, feature, measure, morphology
import scipy.ndimage as ndi
from tqdm import tqdm

def distance_watershed(img, sigma=0.1, coords=None):
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
    return segmentation.watershed(-distance, markers, mask=img).astype(np.uint8)

def segmentFissions(img, fission_props=None, sigma=0):
    """Segmentation of processed fissions using original fission sites as markers for distace Watershed."""
    if fission_props is None:
        return distance_watershed(img, sigma=sigma)
    
    if len(fission_props)>0:
        coords = (fission_props['weighted_centroid-0'], fission_props['weighted_centroid-1'])
        return distance_watershed(img, sigma=sigma, coords=coords)
    return np.zeros(img.shape, dtype=np.uint8)

def segmentFissionsStack(stack, fission_props=None, **kwargs):
    """Iterates segmentFissions."""
    if stack.ndim==2:
        return segmentFissions(stack, fission_props=fission_props, **kwargs).astype(int)
    
    nb_img = stack.shape[0]
    labels = np.zeros(stack.shape, dtype=np.uint8)
    if fission_props is not None:
        for i in tqdm(range(nb_img), total=nb_img):
            labels[i] = segmentFissions(stack[i], fission_props[i], **kwargs)
        
    else:
        for i in tqdm(range(nb_img), total=nb_img):
            labels[i] = segmentFissions(stack[i], **kwargs)
    return labels

def fissionCoords(labels, img):
    if np.any(labels!=0): 
        fission_props = measure.regionprops_table(labels, intensity_image=img, properties=['weighted_centroid'])    
        return (fission_props['weighted_centroid-0'].round().astype(int),
                fission_props['weighted_centroid-1'].round().astype(int))
    else:
        return None

def fissionCoordsStack(labels, stack):
    """Iterates fissionCoords."""
    if labels.ndim==2:
        return fissionCoords(labels, stack)
    
    nb_img = labels.shape[0]
    fission_props = [{}]*nb_img
    for i in tqdm(range(nb_img), total=nb_img):
        fission_props[i] = fissionCoords(labels[i], stack[i])
    return fission_props

def analyzeFissions(labels, img):
    """Find fission sites and measure their diameter"""
    fission_props = {}
    if np.any(labels!=0): 
        fission_props = measure.regionprops_table(labels, intensity_image=img, properties=['weighted_centroid', 'equivalent_diameter'])    
        fission_props['weighted_centroid-0'] = fission_props['weighted_centroid-0'].round().astype(int)
        fission_props['weighted_centroid-1'] = fission_props['weighted_centroid-1'].round().astype(int)
    return fission_props
    
def analyzeFissionsStack(labels, stack):
    """Iterates analyzeFissions."""
    if labels.ndim==2:
        return analyzeFissions(labels, stack)
    
    nb_img = labels.shape[0]
    fission_props = [{}]*nb_img
    for i in tqdm(range(nb_img), total=nb_img):
        fission_props[i] = analyzeFissions(labels[i], stack[i])
    return fission_props

def filterLabels(labels, labels_to_keep):
    """Keep only those labels that are in labels_to_keep."""
    labels_proc = np.zeros(labels.shape, dtype=np.uint8)
    for label in labels_to_keep:
        mask = labels==label
        labels_proc[mask] = labels[mask]
    return labels_proc

def filterLabelsStack(labels, labels_to_keep):
    """Iterates filterLabels."""
    if labels.ndim==2:
        return filterLabels(labels, labels_to_keep)
    
    labels_proc = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(labels.shape[0]):
        labels_proc[i] = filterLabels(labels[i], labels_to_keep[i])
    return labels_proc
    

def prepareProc(img, sigma=1, dilation_nb_sigmas=2, threshold=0, coords=None, mode='same'):
  """Smoothed probability map of divisions. 
  Modes
  - 'same': every fission is assigned the same gaussian intensity profile
  - 'max': every fission has a different gaussian profile, scaled by its maximum intensity"""
    
  mask = segmentation.clear_border(img>0) #Remove objects in contact with the border
  img_proc = img*mask
  labels = distance_watershed(img_proc, sigma=0.1*sigma, coords=coords)
  labels = morphology.remove_small_objects(labels, 9) #Remove too small objects
  img_proc = np.zeros(img.shape, dtype=np.float32)
  if np.any(labels!=0): 
    fission_props = measure.regionprops_table(labels, intensity_image=img, properties=['label', 'max_intensity'])
    labels_to_keep = fission_props['label'][fission_props['max_intensity']>threshold]
    if len(labels_to_keep)>0:
        labels = filterLabels(labels, labels_to_keep)
        fission_props = measure.regionprops_table(labels, intensity_image=img, properties=['weighted_centroid', 'equivalent_diameter'])
        fission_props['weighted_centroid-0'] = fission_props['weighted_centroid-0'].round().astype(int)
        fission_props['weighted_centroid-1'] = fission_props['weighted_centroid-1'].round().astype(int)
        #Add a dot in each fission site
        fissions_coords = (fission_props['weighted_centroid-0'], fission_props['weighted_centroid-1'])
        if mode=='same':
            img_proc[fissions_coords] = 1 
        elif mode=='max':
            img_proc[fissions_coords] = img[fissions_coords] 
        else:
            raise ValueError("'%s' is not a supported mode. Available modes are 'same', 'max'"%mode)
        #Increase minimum spot size
        dilation_radius = round(dilation_nb_sigmas*sigma)
        mask = morphology.binary_dilation(img_proc, morphology.disk(dilation_radius)) 
        #Intensity smoothing
        img_proc = filters.gaussian(img_proc, sigma, truncate=4+dilation_radius)*mask 
        #Normalization
        img_proc = (img_proc-img_proc.min())/(img_proc.max()-img_proc.min())
        return img_proc, fission_props
  return img_proc, {}


def prepareStack(stack, **kwargs):
  """Apply prepareProc to all of the images in stack"""
  if stack.ndim==2:
    return prepareProc(stack, **kwargs)
  
  stack_proc = np.zeros(stack.shape, dtype=np.float32)
  fission_props = [0]*stack.shape[0]
  for i, img in tqdm(enumerate(stack), total=stack.shape[0]):
    stack_proc[i], fission_props[i] = prepareProc(img, **kwargs)
  return stack_proc, fission_props
