# utils/losses.py
import tensorflow as tf

class MDUSTLoss(tf.keras.losses.Loss):
    """
    Composite Loss for M-DUST: L_total = L_cls + lambda_1 * L_seg + lambda_2 * L_con
    """
    def __init__(self, lambda_1=1.0, lambda_2=0.5, name="mdust_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cls_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.seg_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
    def call(self, y_true, y_pred):
        # TensorFlow passes targets and predictions as dictionaries if compiled with multiple outputs
        y_true_cls = y_true['cls']
        y_pred_cls = y_pred['cls']
        
        y_true_seg = y_true['seg']
        y_pred_seg = y_pred['seg']
        
        # Classification Loss
        l_cls = self.cls_loss_fn(y_true_cls, y_pred_cls)
        
        # Segmentation Loss
        l_seg = self.seg_loss_fn(y_true_seg, y_pred_seg)
        
        # Note: Contrastive loss is set to 0.0 here. To implement full contrastive loss, 
        # the model needs to output the pre-classification embeddings alongside cls and seg.
        l_con = 0.0 
        
        return l_cls + (self.lambda_1 * l_seg) + (self.lambda_2 * l_con)