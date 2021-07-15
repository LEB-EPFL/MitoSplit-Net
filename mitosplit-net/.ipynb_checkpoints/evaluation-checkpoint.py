import numpy as np
from tqdm.notebook import tqdm

import tensorflow as tf

def get_metrics(y_true, y_pred, thresholds):
  metrics = {'binary accuracy': tf.metrics.BinaryAccuracy(),
             'precision': tf.metrics.Precision(),
             'recall': tf.metrics.Recall()}
  nb_thresholds = len(thresholds)
  metrics_test = {metric_name: np.zeros(nb_thresholds) for metric_name in metrics}
  y_true_binary = y_true>0
  for i, thr in tqdm(enumerate(thresholds), total=nb_thresholds):
    y_pred_binary = y_pred>=thr
    for metric_name in metrics:
      metrics[metric_name].update_state(y_true_binary, y_pred_binary)
      metrics_test[metric_name][i] = metrics[metric_name].result().numpy()
  return metrics_test

def detection_match(y_true, y_pred, threshold=0.99):
  total = y_true.shape[0]
  true_det = np.any(np.any(y_true, axis=1), axis=1)
  pred_det = np.any(np.any(y_pred>=threshold, axis=1), axis=1)
  return np.sum(np.equal(true_det, pred_det))/total