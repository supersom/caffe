#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import scipy.io
import scipy.misc
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import time

import caffe
from numpy import mod


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default= '', #os.path.join(pycaffe_dir,
                     #        'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='0', #'2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
        if 'inputs[0].shape[2]' not in locals():
            inputs = np.tile(inputs[:,:,:,np.newaxis],(1,1,1,1))
        for idx in range(inputs.shape[0]):
            if not mod(idx,100):
                plt.imshow(np.tile(inputs[idx],(1,1,3)))
                plt.show()
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        print("Loading file: %s" % args.input_file)
        input_ = caffe.io.load_image(args.input_file)
        input_ = scipy.misc.imresize(input_,(28,28),'bilinear')
        plt.imshow(input_)
        plt.show()
        if input_.shape[2] == 3:
            input_ = rgb2gray(input_)
            input_ = np.tile(input_[:,:,np.newaxis], (1,1,1))
        inputs = [input_]

    print("Classifying %d inputs." % len(inputs))

    # Check if inputs have been loaded correctly
    #print '%s' % np.size(inputs[0])
    
    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not args.center_only)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, predictions)
    scipy.io.savemat(args.output_file, mdict={'mnist_test_pred':predictions})


if __name__ == '__main__':
    main(sys.argv)
