from pathlib import Path
import tensorflow as tf
import evaluation
import numpy as np
import tifffile

def adjust_tf_dimensions(stack:np.array):
    if len(stack.shape) < 4:
        return np.expand_dims(stack, axis=-1)
    else:
        return np.moveaxis(stack, 1, -1)

@tf.keras.utils.register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2 * intersection + smooth) / (denominator + smooth)


def main(model_dir: Path = None):
    training_folder = model_dir.parent
    model = tf.keras.models.load_model(model_dir)

    eval_images = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_images_00.tif"))
    frames = min(eval_images.shape[0], 300)
    eval_images = eval_images[:frames]
    eval_mask = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_gt_00.tif"))[:frames]

    pred_output_test = evaluation.predict(eval_images, model)
    labels = evaluation.label(pred_output_test)
    labels = np.expand_dims(labels, axis=-1)
    true_labels = evaluation.label(eval_mask, eval_mask.max() / 2)

    stats = evaluation.fissionStatsStack(true_labels, labels)
    # [TP, FP, FN, TP_px, FP_px, FN_px]
    precision = evaluation.get_precision(stats[0], stats[1])
    tpr = evaluation.get_tpr(stats[0], stats[2])
    f1 = round(evaluation.get_f1_score(precision, tpr)*100)/100
    precision = round(precision*100)/100
    tpr = round(tpr*100)/100
    print(f"FP {stats[1]}")
    print(f"TP {stats[0]}")
    print(f"FN {stats[2]}")
    print(f"TPR {tpr}")
    print(f"Precision {precision}")
    print(f"F1 {f1}")


model_dir = "W:/deep_events/data/original_data/training_data/20230705_1713_brightfield_cos7/20230705_1715_model.h5"

if __name__ == "__main__":
    main(Path(model_dir))