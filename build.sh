#!/usr/bin/env bash 
set -xe
IMAGE_NAME=${IMAGE_NAME:-caffe}
docker build -t ${IMAGE_NAME} .
