# models/layers.py
import tensorflow as tf
from tensorflow.keras import layers

class MSEEMLayer(layers.Layer):
    """Multi-Scale Edge Enhancement Module using the Sobel operator."""
    def __init__(self, **kwargs):
        super(MSEEMLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Extract gradients
        sobel_edges = tf.image.sobel_edges(inputs)
        sobel_y = sobel_edges[..., 0]
        sobel_x = sobel_edges[..., 1]
        
        # Calculate gradient magnitude
        magnitude = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y) + 1e-8)
        return magnitude

class FrequencyFilterLayer(layers.Layer):
    """Applies 2D FFT, a learnable frequency filter, and 2D IFFT."""
    def __init__(self, **kwargs):
        super(FrequencyFilterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.freq_weight = self.add_weight(
            name='freq_weight',
            shape=(1, input_shape[3], input_shape[1], input_shape[2]), # Shape: [1, C, H, W]
            initializer='ones',
            trainable=True
        )
        super(FrequencyFilterLayer, self).build(input_shape)

    def call(self, inputs):
        x_complex = tf.cast(inputs, tf.complex64)
        x_transposed = tf.transpose(x_complex, perm=[0, 3, 1, 2])
        
        # 2D FFT
        fft = tf.signal.fft2d(x_transposed)
        
        # Apply learnable filter
        filter_complex = tf.cast(self.freq_weight, tf.complex64)
        filtered_fft = fft * filter_complex
        
        # 2D IFFT
        ifft = tf.signal.ifft2d(filtered_fft)
        
        # Transpose back and return real part
        ifft_transposed = tf.transpose(ifft, perm=[0, 2, 3, 1])
        return tf.cast(tf.math.real(ifft_transposed), tf.float32)

class DAIMBlock(layers.Layer):
    """Detail-Aware Interaction Module (Cross-Attention)."""
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(DAIMBlock, self).__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, spatial_features, residual_features):
        norm_spatial = self.norm1(spatial_features)
        norm_residual = self.norm2(residual_features)
        
        # Spatial queries attending to Residual (CRI) keys/values
        attention_output = self.attention(
            query=norm_spatial,
            value=norm_residual,
            key=norm_residual
        )
        return spatial_features + attention_output