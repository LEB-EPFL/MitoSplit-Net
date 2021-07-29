import numpy as np
from tqdm import tqdm
import tensorflow as tf

def predict(input_test, model):
    try:
        return model.predict(input_test)[:, :, :, 0]
    except:
        nb_models = len(model)
        pred_output_test = np.zeros((nb_models, *input_test.shape))
        for model_id, model_name in enumerate(models):
            pred_output_test[model_id] = model[model_name].predict(input_test)[:, :, :, 0]
        return pred_output_test
      
def confusion_matrix(outputs, predictions, threshold):
  out_binary = outputs>0
  nb_pixels = out_binary.size

  if isinstance(threshold, (int, float)):
    conf_matrix = np.zeros((2, 2))
    pred_binary = predictions>threshold
    
    tp_mask = out_binary & pred_binary
    fn_mask = out_binary & (~pred_binary)
    fp_mask = (~out_binary) & pred_binary
    
    conf_matrix[0, 0] = tp_mask.sum() #True Positives
    conf_matrix[0, 1] = fn_mask.sum() #False Negatives
    conf_matrix[1, 0] = fp_mask.sum() #False Positives
    conf_matrix[1, 1] = nb_pixels - conf_matrix[0, 0] - conf_matrix[0, 1] - conf_matrix[1, 0] #True Negatives
    return conf_matrix.astype(int)
  
  nb_thr = len(threshold)
  conf_matrix = np.zeros((nb_thr, 2, 2))
  for i, thr in tqdm(enumerate(threshold), total=nb_thr):
    pred_binary = predictions>thr
    
    tp_mask = out_binary & pred_binary
    fn_mask = out_binary & (~pred_binary)
    fp_mask = (~out_binary) & pred_binary
    
    conf_matrix[i, 0, 0] = tp_mask.sum() #True Positives
    conf_matrix[i, 0, 1] = fn_mask.sum() #False Negatives
    conf_matrix[i, 1, 0] = fp_mask.sum() #False Positives
    conf_matrix[i, 1, 1] = nb_pixels - conf_matrix[i, 0, 0] - conf_matrix[i, 0, 1] - conf_matrix[i, 1, 0] #True Negatives
    
  return conf_matrix.astype(int)

def get_metrics(outputs, predictions, threshold):
    conf_matrix = confusion_matrix(outputs, predictions, threshold)
    
    if isinstance(threshold, (int, float)):
        tp = conf_matrix[0, 0]
        fn = conf_matrix[0, 1]
        fp = conf_matrix[1, 0]
        tn = conf_matrix[1, 1]
    else:
        tp = conf_matrix[:, 0, 0]
        fn = conf_matrix[:, 0, 1]
        fp = conf_matrix[:, 1, 0]
        tn = conf_matrix[:, 1, 1]
    
    metrics = {'binary accuracy': (tp+tn)/(tp+tn+fp+fn), 
               'precision': tp/(tp+fp), 
               'TPR': tp/(tp+fn),
               'FPR': fp/(fp+tn)}
    metrics['F1-score'] = 2*(metrics['precision']*metrics['TPR'])/(metrics['precision']+metrics['TPR'])
    return metrics

def detection_match(y_true, y_pred, threshold=0.99):
    total = y_true.shape[0]
    isTrue = np.any(np.any(y_true, axis=-1), axis=-1)
    isPred = np.any(np.any(y_pred>=threshold, axis=1), axis=1)
    return np.sum(np.equal(isPred[None, :], isTrue), axis=1)/total

