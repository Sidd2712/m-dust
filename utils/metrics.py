# utils/metrics.py
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

class F1Score(tf.keras.metrics.Metric):
    """
    Custom Keras Metric to calculate F1-Score for binary classification.
    Crucial for deepfake detection where Real/Fake distributions might be imbalanced.
    """
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(tp, tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(fp, tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(fn, tf.float32)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


class SegmentationIoU(tf.keras.metrics.Metric):
    """
    Calculates the Intersection over Union (IoU) for the predicted forgery masks.
    """
    def __init__(self, name='iou', threshold=0.5, **kwargs):
        super(SegmentationIoU, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > self.threshold, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return self.intersection / (self.union + K.epsilon())

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


def plot_training_history(history, save_path='training_history.png'):
    """
    Utility function to generate clean plots for your MTP presentation.
    Plots Classification Loss, Segmentation Loss, and AUC.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Total Loss
    axes[0].plot(history.history['loss'], label='Train Total Loss')
    axes[0].plot(history.history['val_loss'], label='Val Total Loss')
    axes[0].set_title('Total Network Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    # Plot AUC
    axes[1].plot(history.history['cls_auc'], label='Train AUC')
    axes[1].plot(history.history['val_cls_auc'], label='Val AUC')
    axes[1].set_title('Classification AUC')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

    # Plot F1 Score (If you use it)
    if 'cls_f1_score' in history.history:
        axes[2].plot(history.history['cls_f1_score'], label='Train F1')
        axes[2].plot(history.history['val_cls_f1_score'], label='Val F1')
        axes[2].set_title('Classification F1-Score')
        axes[2].set_xlabel('Epochs')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history plot saved to {save_path}")