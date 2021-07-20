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
    isTrue = np.any(np.any(y_true, axis=-1), axis=-1)
    isPred = np.any(np.any(y_pred>=threshold, axis=1), axis=1)
    return np.sum(np.equal(isPred[None, :], isTrue), axis=1)/total

