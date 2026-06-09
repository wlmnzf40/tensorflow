#!/usr/bin/env python3
"""
Generate a GEMM (Matrix Multiply) SavedModel for TF Serving.

Usage:
  # Direct (if TF Python is working)
  python3 generate_model.py

  # Via Docker (recommended when local TF install is broken)
  python3 generate_model.py --docker
"""
import argparse
import os
import subprocess
import sys


def generate_via_docker(output_dir: str):
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)

    # Inline the model-generation code and run it inside the official TF image.
    model_code = """
import tensorflow as tf

class GEMMModel(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.float32, name='A'),
        tf.TensorSpec([None, None], tf.float32, name='B'),
    ])
    def __call__(self, A, B):
        return {'output': tf.linalg.matmul(A, B)}

model = GEMMModel()
tf.saved_model.save(model, '/export/1')
print('Model saved to /export/1')
"""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_dir}:/export",
        "tensorflow/tensorflow:2.15.0",
        "python3", "-c", model_code,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done. SavedModel written to: {abs_dir}/1")


def generate_direct(output_dir: str):
    try:
        import tensorflow as tf
    except ImportError as e:
        sys.exit(
            f"TensorFlow not importable: {e}\n"
            "Use --docker to generate via Docker instead."
        )

    class GEMMModel(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([None, None], tf.float32, name='A'),
            tf.TensorSpec([None, None], tf.float32, name='B'),
        ])
        def __call__(self, A, B):
            return {'output': tf.linalg.matmul(A, B)}

    model = GEMMModel()
    save_path = os.path.join(output_dir, '1')
    tf.saved_model.save(model, save_path)
    print(f"Done. SavedModel written to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='gemm_model',
                        help='Output directory (a "1/" sub-dir is created)')
    parser.add_argument('--docker', action='store_true',
                        help='Use the official TF Docker image to generate the model')
    args = parser.parse_args()

    if args.docker:
        generate_via_docker(args.output)
    else:
        generate_direct(args.output)
