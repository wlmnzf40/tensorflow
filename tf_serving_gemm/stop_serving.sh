#!/bin/bash
docker rm -f tf-serving-gemm 2>/dev/null && echo "Stopped tf-serving-gemm" || echo "Container not running"
