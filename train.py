import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from models.mdust import build_mdust_model
from utils.losses import MDUSTLoss
from utils.metrics import F1Score, SegmentationIoU, plot_training_history

# 1. Hyperparameters 
BATCH_SIZE = 32
INITIAL_LR = 1e-4
EPOCHS = 30

# 2. Prepare Datasets (Assuming c23 split for FF++)
train_ds = build_dataset('data/FF++_c23/train', batch_size=BATCH_SIZE)
val_ds = build_dataset('data/FF++_c23/val', batch_size=BATCH_SIZE)

# 3. Learning Rate Scheduler (Cosine Annealing) [cite: 355]
decay_steps = 1000 # Adjust based on dataset size: (num_samples // batch_size) * epochs
lr_schedule = CosineDecay(
    initial_learning_rate=INITIAL_LR,
    decay_steps=decay_steps,
    alpha=0.0 # Minimum learning rate multiplier
)

# 4. Initialize Optimizer and Model [cite: 355]
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
model = build_mdust_model()

# 5. Compile Model with Composite Loss
model.compile(
    optimizer=optimizer,
    loss=MDUSTLoss(lambda_1=1.0, lambda_2=0.5),
    metrics={
        'cls': ['accuracy', tf.keras.metrics.AUC(name='auc'), F1Score()],
        'seg': ['accuracy', SegmentationIoU()]
    }
)

# 6. Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='mdust_best_weights.keras',
        monitor='val_cls_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_cls_auc', patience=5, mode='max')
]

# 7. Execute Training
print("Initiating M-DUST Training Pipeline...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

plot_training_history(history, save_path='mdust_training_results.png')