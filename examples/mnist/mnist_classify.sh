#!/usr/bin/env sh
IMAGES_PATH=data/mnist/mnist_testing_images.npy
OUT_FILE=mnist_test_pred

MODEL_DEF=examples/mnist/lenet.prototxt
MODEL=examples/mnist/lenet_iter_10000.caffemodel

echo "MNIST data - testing start"
cd ~/code/caffe
python python/classify_mnist.py --model_def $MODEL_DEF --pretrained_model $MODEL --center_only --images_dim 28,28 $IMAGES_PATH $OUT_FILE
echo "MNIST data - testing done!"