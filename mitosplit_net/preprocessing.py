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
    """Returns weighted centroids of fissions from labels"""
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
    

def prepareProc(img, sigma=1, dilation_nb_sigmas=2, threshold=0, coords=None, mode='max', vmax=255):
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
        #Fission probability thresholding
        labels_to_keep = fission_props['label'][fission_props['max_intensity']>threshold]
        if len(labels_to_keep)>0:
            labels = filterLabels(labels, labels_to_keep)
            fission_props = measure.regionprops_table(labels, intensity_image=img, properties=['weighted_centroid', 'equivalent_diameter'])
            fission_props['weighted_centroid-0'] = fission_props['weighted_centroid-0'].round().astype(int)
            fission_props['weighted_centroid-1'] = fission_props['weighted_centroid-1'].round().astype(int)
            #Add a dot in each fission site
            fission_coords = (fission_props['weighted_centroid-0'], fission_props['weighted_centroid-1'])
            if mode=='same':
                img_proc[fission_coords] = 1 
            elif mode=='max':
                for y, x in zip(*fission_coords):
                    roi_mask = labels==labels[y, x]
                    img_proc[y, x] = img[roi_mask].max()   
            else:
                raise ValueError("'%s' is not a supported mode. Available modes are 'same', 'max'"%mode)
            #Increase minimum spot size
            dilation_radius = round(dilation_nb_sigmas*sigma)
            mask = morphology.binary_dilation(img_proc, morphology.disk(dilation_radius)) 
            #Intensity smoothing
            img_proc = filters.gaussian(img_proc, sigma, truncate=4+dilation_radius)*mask 
            #Normalization
            img_proc = img_proc*img.max()/img_proc.max()/vmax
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

def track(fission_props, num_diam=1, min_event_duration=3):
    """Returns a list of trajectories.
    Parameters
    ----------
    fission_props: list
        List of dictionaries with the position and diameter of each fission per frame
    
    num_diam:
    
    Return
    T: list
        List of fission site trajectories. Each element includes a tuple with (initial time, final time) and an array with the sequence of positions.
    """
    track_props = []

    for f in fission_props:
        if len(f)>0:
            track_props += [np.array(tuple(zip(f['weighted_centroid-0'], f['weighted_centroid-1'], f['equivalent_diameter'])), dtype=np.int8)]
        else:
            track_props += [[]]

    #List of trajectories
    T = []

    num_frames = len(track_props)
    for t0 in range(num_frames-1):
        source = track_props[t0]
        for src in source:
            #Initialize trajectory and fission size
            x_src = src[:2]
            T += [x_src[None, :]]
            sigma_src = src[2] 
            for t1 in range(t0+1, num_frames):
                target = track_props[t1] #List of possible new positions

                if len(target)>0:
                    #Find nearest target
                    x_target = target[:, :2]
                    distance = np.linalg.norm(x_target-x_src, axis=1)/sigma_src #Distance between source (t0) and target (t1)
                    i_target = np.argmin(distance)

                    #If nearest target overlaps with src, add it to the trajectory and remove it from track_props.
                    #Else, end trajectory.
                    if distance[i_target]<num_diam:
                        #Update position and sigma
                        x_src = x_target[i_target]
                        sigma_src = target[i_target, 2]
                        
                        T[-1] = np.append(T[-1], x_src[None, :], axis=0)
                        track_props[t1] = np.delete(track_props[t1], i_target, axis=0)
                    else:
                        break
                #End trajectory if there are no fission sites in the next frame
                else:
                    break
            #Save initial and final time
            event_duration = len(T[-1])
            if event_duration<min_event_duration:
                T.pop(-1)
            else:                
                T[-1] = ((t0, t0+event_duration), T[-1])
    
    return T

def get_event_score(output, labels, T):
    """Returns maximum event score of each fission over time."""
    event_score = []
    time = []
    track_labels = []
    for t_bounds, centroids in T:
        t = np.arange(*t_bounds)
        time += [t]
        event_score += [[]]
        track_labels += [[]]
        for output_img, label_img, x in zip(output[t], labels[t], centroids):
            label = label_img[x[0], x[1]]
            roi_mask = label_img==label
            track_labels[-1] += [label]
            event_score[-1] += [output_img[roi_mask].max()]
        event_score[-1] = (255*np.array(event_score[-1])).astype(np.uint8)
        track_labels[-1] = np.array(track_labels[-1]).astype(np.uint8)
    return time, track_labels, event_score

def _selFissions(output, labels, new_output, new_labels, track_time, track_labels, add_signal):
    if np.any(add_signal):
        time_add = track_time[add_signal]
        labels_add = track_labels[add_signal]
        for t, roi in zip(time_add, labels_add):
            roi_mask = labels[t]==roi
            new_output[t][roi_mask] = output[t][roi_mask]
            new_labels[t][roi_mask] = roi
    return new_output, new_labels

def selFissions(output, labels, track_time, track_labels, event_score, threshold=100):    
    new_labels = np.zeros_like(labels)
    new_output = np.zeros_like(output)
    for time, lab, score in zip(track_time, track_labels, event_score):
        isHigh = (score>=threshold)
        new_output, new_labels = _selFissions(output, labels, new_output, new_labels, time, lab, isHigh)
    return new_output, new_labels