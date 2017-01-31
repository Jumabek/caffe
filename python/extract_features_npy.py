import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', metavar='path_to_caffe', type=str, default='caffe/',
                    help='path to Caffe')
parser.add_argument('-p', metavar='path_to_img', type=str, default='images/',
                    help='path to image files')
parser.add_argument('-i', metavar='image_filenames', type=str, default='filenames.npy',
                    help='npy filename containing image filenames')
parser.add_argument('-o', metavar='features_filename', type=str, default='features.npy',
                    help='npy filename wirtes extracted features in')
parser.add_argument('-g', metavar='gpu', type=int, default=-1,
                    help='GPU device ID (CPU if this negative)')
args = parser.parse_args()

caffe_root = args.c
path_to_img = args.p
mean        = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
deploy      = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model       = caffe_root + 'models/bvlc_reference_caffenet/bvlc_alexnet.caffemodel'
feat_layer = 'pool5'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
if args.g < 0:
    caffe.set_mode_cpu()
else:
    caffe.set_device(args.g)
    caffe.set_mode_gpu()

net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

images = np.load(args.i)
N = len(images)
net.blobs['data'].reshape(N,3,227,227)
for i in range(N):
    
    net.blobs['data'].data[i] = \
        transformer.preprocess('data', caffe.io.load_image(images[i]))
net.forward()
print net.blobs[feat_layer].data.shape
np.save(args.o, net.blobs[feat_layer].data)
