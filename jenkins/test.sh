#!/usr/bin/env bash
set -xe
IMAGE_NAME=${IMAGE_NAME:-caffe}
cd `git rev-parse --show-toplevel`
docker run --rm -v $PWD:/workspace -w /workspace ${IMAGE_NAME} bash -xec "python setup.py test"
