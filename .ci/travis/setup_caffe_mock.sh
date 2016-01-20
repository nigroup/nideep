#!/bin/bash
set -e # exit with nonzero exit code if anything fails

ROOT_DIR=$TRAVIS_BUILD_DIR
if [[ -z "$ROOT_DIR" ]]; then
    echo "ROOT_DIR not set!"
    exit 1
fi;
    
mkdir -p $ROOT_DIR/caffe/proto/
protoc -I=$ROOT_DIR/ --python_out=$ROOT_DIR/caffe/proto/ $ROOT_DIR/caffe.proto

# For mocking caffe
mv $ROOT_DIR/caffe_mock.py $ROOT_DIR/caffe/__init__.py
touch $ROOT_DIR/caffe/proto/__init__.py
