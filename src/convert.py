"""TFLite conversion utilities and C-array header export."""

import os
import gzip
import numpy as np
import tensorflow as tf

def convert_to_tflite_fp32(keras_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved FP32 TFLite: {output_path} ({len(tflite_model) / 1024:.1f} KB)")
    return tflite_model

def convert_to_tflite_fp16(keras_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved FP16 TFLite: {output_path} ({len(tflite_model) / 1024:.1f} KB)")
    return tflite_model

def convert_to_tflite_int8(keras_model, representative_data, output_path, is_audio=False):
    """
    Full INT8 integer post-training quantization.
    representative_data: numpy array of samples.
    """
    def rep_dataset():
        for i in range(min(200, len(representative_data))):
            sample = representative_data[i:i+1].astype(np.float32)
            yield [sample]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved Full INT8 TFLite: {output_path} ({len(tflite_model) / 1024:.1f} KB)")
    return tflite_model

def export_c_array(tflite_path, header_path, array_name):
    """
    Exports a .tflite model binary as a C byte array header file.
    Cross-platform equivalent to 'xxd -i'.
    """
    with open(tflite_path, "rb") as f:
        bytes_data = f.read()
        
    num_bytes = len(bytes_data)
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    
    with open(header_path, "w") as f:
        f.write(f"// Generated C-array for {os.path.basename(tflite_path)}\n")
        f.write(f"// Size: {num_bytes} bytes\n\n")
        f.write("#ifndef " + array_name.upper() + "_H\n")
        f.write("#define " + array_name.upper() + "_H\n\n")
        f.write(f"alignas(16) const unsigned char {array_name}[] = {{\n  ")
        
        for i, b in enumerate(bytes_data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
                
        f.write("\n};\n\n")
        f.write(f"const unsigned int {array_name}_len = {num_bytes};\n\n")
        f.write("#endif // " + array_name.upper() + "_H\n")
        
    print(f"Exported C-array header to {header_path} ({num_bytes} bytes)")

def get_file_sizes_kb(file_path):
    raw_size = os.path.getsize(file_path) / 1024.0
    with open(file_path, "rb") as f:
        data = f.read()
    gzip_data = gzip.compress(data)
    gzip_size = len(gzip_data) / 1024.0
    return raw_size, gzip_size
