#!/bin/bash
# Start TF Serving via Docker.
# gRPC port: 8500    REST port: 8501

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/gemm_model"
CONTAINER_NAME="tf-serving-gemm"

if [ ! -d "${MODEL_DIR}/1" ]; then
    echo "[ERROR] Model not found at ${MODEL_DIR}/1"
    echo "        Run: python3 generate_model.py [--docker] first"
    exit 1
fi

# Remove stale container if it exists.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}"
fi

echo "Starting TF Serving ..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    -p 8500:8500 \
    -p 8501:8501 \
    -v "${MODEL_DIR}:/models/gemm" \
    -e MODEL_NAME=gemm \
    tensorflow/serving:2.15.0

echo ""
echo "TF Serving started:"
echo "  gRPC -> localhost:8500"
echo "  REST -> http://localhost:8501"
echo ""
echo "Check model status:"
echo "  curl http://localhost:8501/v1/models/gemm"
echo ""
echo "Stop with: ./stop_serving.sh"
