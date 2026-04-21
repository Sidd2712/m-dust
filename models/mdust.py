from tensorflow.keras import layers, Model
from .layers import MSEEMLayer, FrequencyFilterLayer, DAIMBlock

def build_mdust_model(input_shape=(224, 224, 3)):
    # 1. Input Definitions
    rgb_input = layers.Input(shape=input_shape, name='rgb_input')
    cri_input = layers.Input(shape=input_shape, name='cri_input') # Computed in data pipeline
    
    # --- STREAM 1: Spatial-Edge Stream ---
    edge_features = MSEEMLayer()(rgb_input)
    # Patch partition & Transformer Encoder (Simplified via Conv2D for feature extraction)
    spatial_encoded = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(edge_features)
    spatial_encoded = layers.MaxPooling2D((2, 2))(spatial_encoded)
    
    # --- STREAM 2: Color-Residual Stream ---
    cri_encoded = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(cri_input)
    cri_encoded = layers.MaxPooling2D((2, 2))(cri_encoded)
    
    # --- FUSION 1: Detail-Aware Interaction Module (DAIM) ---
    # Flatten spatial and CRI features for multi-head attention
    # Reshape to (Batch, Sequence_Length, Features)
    seq_len = spatial_encoded.shape[1] * spatial_encoded.shape[2]
    spatial_flat = layers.Reshape((seq_len, 64))(spatial_encoded)
    cri_flat = layers.Reshape((seq_len, 64))(cri_encoded)
    
    fused_spatial_cri = DAIMBlock(embed_dim=64, num_heads=4)(spatial_flat, cri_flat)
    
    # --- STREAM 3: Frequency Stream ---
    freq_features = FrequencyFilterLayer()(rgb_input)
    freq_encoded = layers.Conv2D(64, (3, 3), strides=(4, 4), padding='same', activation='relu')(freq_features)
    freq_flat = layers.Reshape((seq_len, 64))(freq_encoded)
    
    # --- FUSION 2: Cross-Modality Fusion (CMF) ---
    # Fusing the (Spatial+CRI) with Frequency features
    final_features = layers.Concatenate(axis=-1)([fused_spatial_cri, freq_flat])
    final_features = layers.GlobalAveragePooling1D()(final_features)
    
    # --- HEADS ---
    # Classification Head (Real/Fake)
    x_cls = layers.Dense(128, activation='relu')(final_features)
    x_cls = layers.Dropout(0.3)(x_cls)
    output_cls = layers.Dense(1, activation='sigmoid', name='cls')(x_cls)
    
    # Segmentation Head (Forgery Mask)
    # Re-expand dimensions to generate an image mask
    x_seg = layers.Dense(56 * 56 * 32, activation='relu')(final_features)
    x_seg = layers.Reshape((56, 56, 32))(x_seg)
    x_seg = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x_seg)
    x_seg = layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(x_seg)
    output_seg = layers.Conv2D(1, (1, 1), activation='sigmoid', name='seg')(x_seg)
    
    # Build Model
    model = Model(inputs=[rgb_input, cri_input], outputs=[output_cls, output_seg], name="M-DUST")
    return model