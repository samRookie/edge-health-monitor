"""Model compression suite: Magnitude pruning, FP16, Full INT8 PTQ, and Knowledge Distillation."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import gzip
import yaml
import numpy as np
import tensorflow as tf
import tf_keras as keras

from src.models import build_kws_teacher_model, build_kws_student_model
from src.convert import convert_to_tflite_fp32, convert_to_tflite_fp16, convert_to_tflite_int8, export_c_array, get_file_sizes_kb
from src.data_har import load_har_splits
from src.data_kws import load_kws_splits

def apply_magnitude_pruning(keras_model, X_train, y_train, target_sparsity=0.50, epochs=3):
    """
    Applies magnitude pruning to dense and convolutional layers.
    If tfmot has Keras 3 incompatibilities, falls back gracefully to manual magnitude thresholding.
    """
    print(f"Applying magnitude pruning (Target sparsity: {target_sparsity*100:.0f}%)...")
    try:
        import tensorflow_model_optimization as tfmot
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity, begin_step=0, frequency=100
            )
        }
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(keras_model, **pruning_params)
        pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        pruned_model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        return final_model
    except Exception as e:
        print(f"tfmot pruning warning ({e}). Applying weight masking fallback...")
        cloned_model = keras.models.clone_model(keras_model)
        cloned_model.set_weights(keras_model.get_weights())
        weights = cloned_model.get_weights()
        for i in range(len(weights)):
            # Only prune kernel weights (ndim >= 2), leave bias and norm scales intact
            if weights[i].ndim >= 2:
                thresh = np.percentile(np.abs(weights[i]), target_sparsity * 100)
                mask = np.abs(weights[i]) >= thresh
                weights[i] = weights[i] * mask
        cloned_model.set_weights(weights)
        return cloned_model

class DistillationModel(keras.Model):
    """Knowledge distillation training wrapper combining student and teacher."""
    def __init__(self, student, teacher, temperature=3.0, alpha=0.3):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.distillation_loss_fn = keras.losses.KLDivergence()

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Soft targets
            teacher_soft = tf.nn.softmax(tf.math.log(teacher_predictions + 1e-8) / self.temperature)
            student_soft = tf.nn.softmax(tf.math.log(student_predictions + 1e-8) / self.temperature)
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)
            
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["student_loss"] = student_loss
        return results

def run_distillation(kws_splits, quick=False):
    print("\n=== Running Knowledge Distillation on KWS ===")
    X_train, y_train = kws_splits["train"]
    X_val, y_val = kws_splits["val"]
    X_test, y_test = kws_splits["test"]
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    epochs = 4 if quick else 12
    
    # 1. Train Teacher
    print("Training Teacher Model (~120k params)...")
    teacher = build_kws_teacher_model(input_shape=(98, 40, 1), num_classes=8)
    teacher.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    teacher.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, verbose=0)
    teacher_acc = float(np.mean(np.argmax(teacher.predict(X_test, verbose=0), axis=-1) == y_test))
    print(f"Teacher Test Accuracy: {teacher_acc*100:.2f}%")
    
    # 2. Train Student Without Distillation (baseline)
    print("Training Student Model Without Distillation (~15k params)...")
    student_base = build_kws_student_model(input_shape=(98, 40, 1), num_classes=8)
    student_base.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    student_base.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, verbose=0)
    student_base_acc = float(np.mean(np.argmax(student_base.predict(X_test, verbose=0), axis=-1) == y_test))
    print(f"Student Without Distillation Accuracy: {student_base_acc*100:.2f}%")
    
    # 3. Train Student With Distillation
    print("Training Student Model With Distillation (Soft targets T=3.0, alpha=0.3)...")
    student_dist = build_kws_student_model(input_shape=(98, 40, 1), num_classes=8)
    dist_model = DistillationModel(student=student_dist, teacher=teacher, temperature=3.0, alpha=0.3)
    dist_model.compile(optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    dist_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, verbose=0)
    student_dist_acc = float(np.mean(np.argmax(student_dist.predict(X_test, verbose=0), axis=-1) == y_test))
    print(f"Student With Distillation Accuracy: {student_dist_acc*100:.2f}%")
    
    res = {
        "teacher_params": int(teacher.count_params()),
        "teacher_accuracy": teacher_acc,
        "student_params": int(student_base.count_params()),
        "student_base_accuracy": student_base_acc,
        "student_distilled_accuracy": student_dist_acc,
        "distillation_gain_pct": float((student_dist_acc - student_base_acc) * 100.0)
    }
    with open("results/distillation_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    student_dist.save("models/kws_student_distilled.keras")
    print("Saved results/distillation_metrics.json and models/kws_student_distilled.keras")
    return student_dist
