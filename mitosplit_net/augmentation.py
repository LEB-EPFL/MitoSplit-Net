
import numpy as np
from skimage import segmentation
from albumentations import Compose, Rotate, RandomRotate90, HorizontalFlip, Flip, ElasticTransform, GaussNoise, Crop, Resize
from tqdm import tqdm

def augImg(input_img, output_img, labels, transform, noise_transform, **kwargs):
    input_mask = (input_img>0).astype(np.uint8)
    transformed = transform(image=input_img, image0=output_img, mask=labels, mask0=input_mask)
    transformed['image'] = noise_transform(image=transformed['image'])['image']
    
    aug_input_img, aug_output_img, aug_labels = transformed['image']*transformed['mask0'], transformed['image0'], transformed['mask']
    
    aug_fission_coords = preprocessing.fissionCoords(aug_labels, aug_output_img)
    aug_output_img, aug_fission_props = preprocessing.prepareProc(aug_output_img, coords=aug_fission_coords, **kwargs)
    aug_labels = preprocessing.segmentFissions(aug_output_img, aug_fission_props)
    return aug_input_img.astype(np.uint8), aug_output_img, aug_labels

def augStack(input_data, output_data, labels, transform, noise_transform, **kwargs):
    if input_data.ndim==2:
        return augImg(input_data, output_data, labels, transform, noise_transform, **kwargs)

    aug_input_data = np.zeros(input_data.shape, dtype=np.uint8)
    aug_output_data = np.zeros(output_data.shape, dtype=np.float32)
    aug_labels = np.zeros(labels.shape, dtype=np.uint8)
    
    for i in tqdm(range(input_data.shape[0]), total=input_data.shape[0]):
        aug_input_data[i], aug_output_data[i], aug_labels[i] = augImg(input_data[i], output_data[i], labels[i], transform, noise_transform, **kwargs)
    return aug_input_data, aug_output_data, aug_labels