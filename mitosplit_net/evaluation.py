import numpy as np
from skimage import filters, measure
from tqdm import tqdm
import tensorflow as tf

#Binary classification of predictions (TP, FP and FN)

def predict(input_test, model):
    """Returns model predictions on input test dataset."""
    try:
        return model.predict(input_test)[:, :, :, 0].astype(np.float32)
    except:
        nb_models = len(model)
        pred_output_test = np.zeros((nb_models, *input_test.shape), dtype=np.float32)
        for model_id, model_name in enumerate(model):
            pred_output_test[model_id] = model[model_name].predict(input_test)[:, :, :, 0]
        return pred_output_test

def label(pred_outputs, threshold=None):
    """Performs thresholding on pred_output and labels non-overlapping regions of interest"""
    if threshold is None:
        threshold = filters.threshold_otsu(pred_outputs)
    if isinstance(threshold, (int, np.integer, float, np.floating)):
        if pred_outputs.ndim==2:
            return measure.label(pred_outputs > threshold)
        
        labels = np.zeros(pred_outputs.shape, dtype=np.uint8)
        for i in range(labels.shape[0]):
            labels[i] = measure.label(pred_outputs[i] > threshold)
        return labels
    
    if len(pred_outputs)==len(threshold):
        labels = np.zeros(pred_outputs.shape, dtype=np.uint8)
        for i in range(labels.shape[0]):
            labels[i] = measure.label(pred_outputs[i] > threshold[i])
        return labels
    else:
        raise ValueError("'pred_outputs' and 'threshold' lenghts don't match.")

def fissionStats(true_labels, pred_labels):
    """Calculates TP, FP and FN of predicted areas of interest (pred_labels) by comparing them to the ground truth (true_labels)."""
    TP, FN, FP = 0, 0, 0
    TP_px, FN_px, FP_px = 0, 0, 0
    used_pred_labels = np.zeros(pred_labels.shape, dtype=bool) #Mask of labels
    pred_mask = pred_labels>0
    if np.any(true_labels!=0):
        for true_fission in np.unique(true_labels)[1:]: #Positives, first label is skipped as it corresponds to background
            true_mask = true_labels==true_fission
            overlapping_fissions = np.unique(pred_labels[true_mask]) #All of the pred_labels that overlap with true_fission
            overlapping_fissions = overlapping_fissions[overlapping_fissions!=0] #Remove bg_label
            if len(overlapping_fissions)>0: 
                TP += 1  #Add a TP if there is at least one pred_label that overlaps with the true fission
                TP_px += (true_mask & pred_mask).sum()
                for pred_fission in overlapping_fissions:
                    used_pred_labels[pred_labels==pred_fission] = True #Register used labels in a mask
            else:
                FN += 1 #Add a FN if the true fission was not detected
                FN_px += (true_mask & ~pred_mask).sum()
        remaining_pred_labels = np.unique(pred_labels[~used_pred_labels]) #Keep only pred_labels that are not in contact with true_labels
        remaining_pred_labels = remaining_pred_labels[remaining_pred_labels!=0] #Remove bg_label
        if len(remaining_pred_labels)>0:
            FP = len(remaining_pred_labels) #Add the remaining_pred_labels as FP
            FP_px += (~true_mask & pred_mask).sum()
    else:
        remaining_pred_labels = np.unique(pred_labels[pred_mask])
        if len(remaining_pred_labels)>0:
            FP = len(remaining_pred_labels) #Add the remaining_pred_labels as FP
            FP_px += pred_labels[pred_mask].sum()
    return np.array([TP, FP, FN, TP_px, FP_px, FN_px])

def fissionStatsStack(true_labels, pred_labels):
    """Iterates fissionStats"""
    if true_labels.ndim==2:
        return fissionStats(true_labels, pred_labels)
    
    stats = np.zeros(6, dtype=float)
    for true_lab, pred_lab in zip(true_labels, pred_labels):
        stats += fissionStats(true_lab, pred_lab)
    return stats

#Evaluation metrics and optimal threshold for network predictions as the one that maximizes F1-score
def get_precision(tp, fp): 
    return tp/(tp+fp)
    
def get_tpr(tp, fn):
    return tp/(tp+fn)

def get_f1_score(precision, tpr):
    return 2*(precision*tpr)/(precision+tpr)

def get_fbeta_score(precision, tpr, beta=0.1):
    return (1 + beta**2) * (precision * tpr) / (beta**2 * precision + tpr)

def get_fbeta_curve(true_labels, pred_output, threshold, beta=0.1): 
    fbeta_score = []
    
    for thr in tqdm(threshold, total=len(threshold)):
        pred_labels = label(pred_output, threshold=thr)
        tp, fp, fn = fissionStatsStack(true_labels, pred_labels)[:3]
        precision = get_precision(tp, fp)
        tpr = get_tpr(tp, fn)
        fbeta_score += [get_fbeta_score(precision, tpr, beta)]
    
    return np.array(fbeta_score)

def get_f1_curve(true_labels, pred_output, threshold): 
    f1_score = []
    
    for thr in tqdm(threshold, total=len(threshold)):
        pred_labels = label(pred_output, threshold=thr)
        tp, fp, fn = fissionStatsStack(true_labels, pred_labels)[:3]
        precision = get_precision(tp, fp)
        tpr = get_tpr(tp, fn)
        f1_score += [get_f1_score(precision, tpr)]
    
    return np.array(f1_score)

def get_optimal_threshold(threshold, f1_score):
    return threshold[np.nanargmax(f1_score)]

def detection_match(y_true, y_pred, threshold=None):
    if threshold is None:
        threshold = filters.threshold_otsu(y_pred)
    total = y_true.shape[0]
    isTrue = np.any(np.any(y_true, axis=-1), axis=-1)

    if isinstance(threshold, (int, np.integer, float, np.floating)):
        isPred = np.any(np.any(y_pred>threshold, axis=-1), axis=-1)
        return np.sum(np.equal(isPred, isTrue))/total
    else:  
        nb_thresholds = len(threshold)
        isPred = np.zeros((nb_thresholds, *isTrue.shape), dtype=bool)
        for i in tqdm(range(nb_thresholds), total=nb_thresholds):
            isPred[i] = np.any(np.any(y_pred>threshold[i], axis=-1), axis=-1)
        return np.sum(np.equal(isPred, isTrue), axis=1)/total
