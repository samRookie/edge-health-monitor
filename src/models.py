"""Neural Network architectures for HAR 1D-CNN and KWS DS-CNN.
Includes standard baselines, overfit/underfit variants, and distillation teacher/student models.
"""

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

def build_har_model(input_shape=(128, 6), num_classes=6):
    """
    Compact 1D-CNN for 6-axis IMU Activity Recognition.
    Target: < 50,000 parameters.
    """
    inputs = keras.Input(shape=input_shape, name="imu_input")
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity_probs")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="har_1d_cnn")
    return model

def build_har_underfit_model(input_shape=(128, 6), num_classes=6):
    """Deliberately under-capacity model for Unit 2 overfit/underfit demo."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(4, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="har_underfit")

def build_har_overfit_model(input_shape=(128, 6), num_classes=6):
    """Deliberately over-parameterized model with no regularization."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="har_overfit")

def build_kws_model(input_shape=(98, 40, 1), num_classes=8):
    """
    Depthwise-Separable CNN (DS-CNN) for 8-class Keyword Spotting.
    Receptive field scaled with MaxPooling to cover 1-second keyword utterances.
    Target: < 100,000 parameters.
    """
    inputs = keras.Input(shape=input_shape, name="audio_mfcc_input")
    
    # Block 1: Initial Conv2D + Pool
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2: DS-Conv + Pool
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3: DS-Conv + Pool
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Dense Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="keyword_probs")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="kws_ds_cnn")
    return model

def build_kws_teacher_model(input_shape=(98, 40, 1), num_classes=8):
    """Larger teacher model for knowledge distillation (~120k parameters)."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(48, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    for filters in [64, 96, 128]:
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size=(1, 1), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="kws_teacher")

def build_kws_student_model(input_shape=(98, 40, 1), num_classes=8):
    """Compact student model for knowledge distillation (~15k parameters)."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    for _ in range(2):
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, kernel_size=(1, 1), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="kws_student")

if __name__ == "__main__":
    har = build_har_model()
    print(f"HAR Baseline Parameters: {har.count_params():,}")
    assert har.count_params() < 50000, "HAR model exceeded 50k parameters!"
    
    kws = build_kws_model()
    print(f"KWS Baseline Parameters: {kws.count_params():,}")
    assert kws.count_params() < 100000, "KWS model exceeded 100k parameters!"
    print("Model architectures verified successfully.")
