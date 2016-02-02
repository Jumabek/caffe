#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import lmdb
from numpy import linalg as LA
import caffe
from asyncore import write


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="i.e. train.txt "
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    parser.add_argument(
        "blob",
        help="specifiy the blob you want to extract features from. i.e. fc6"
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
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
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
        default='2,1,0',
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
        mean = np.load(args.mean_file).mean(1).mean(1)
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

#     # Load numpy array (.npy), directory glob (*.jpg), or image file.
#     args.input_file = os.path.expanduser(args.input_file)
#     if args.input_file.endswith('npy'):
#         print("Loading file: %s" % args.input_file)
#         inputs = np.load(args.input_file)
#     elif os.path.isdir(args.input_file):
#         print("Loading folder: %s" % args.input_file)
#         inputs =[caffe.io.load_image(im_f)
#                  for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
#     else:
#         print("Loading file: %s" % args.input_file)
#         inputs = [caffe.io.load_image(args.input_file)]
    
    
    #read input files
    f=open(args.input_file)
    lines=[line.rstrip("\n") for line in f.readlines()]
    im_fs=[line.split()[0] for line in lines]
    labels=[line.split()[1] for line in lines]
    
    print("Example",im_fs[0],labels[0])
    
    num_examples=len(im_fs)
    batch_size=30
    num_batch=int(num_examples/batch_size)
    
    
    
    # one example needs 4000 bytes and I am allocating 10 times bigger one
    map_size=10*num_examples*40000
    
    env=lmdb.open(args.output_file,map_size)
    start = time.time()
    with env.begin(write=True) as txn:           
        for i in range(num_batch):          
                inputs =[caffe.io.load_image(im_f)
                         for im_f in im_fs[i*batch_size:(i+1)*batch_size]] #get i-th batch im_fs 
                
                print("Classifying %d-th batch inputs " % i)
                batch_extraction=classifier.extract(inputs,args.blob, oversample=True) # take the only center crop of the image. Intended to change later on
                #print(batch_extraction.shape)
                for e in range(batch_extraction.shape[0]):
                    batch_extraction[e]=batch_extraction[e]/LA.norm(batch_extraction[e])
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = batch_extraction.shape[1]
                    datum.height = 1
                    datum.width = 1
                    datum.data = batch_extraction[e].tostring()
                    #example id
                    id=i*batch_size+e
                    datum.label=int(labels[id])
                    str_id = '{:08}'.format(id)
                    txn.put(str_id,datum.SerializeToString())
#                 if i==0:
#                     extractions=batch_extraction
#                 else:
#                     extractions=np.concatenate((extractions,batch_extraction),axis=0)
                    
        #last batch. 
        if num_batch*batch_size<num_examples:
            inputs =[caffe.io.load_image(im_f) for im_f in im_fs[num_batch*batch_size:]] #get i-th batch im_fs
            print("Classifying last batch inputs ")
            batch_extraction=classifier.extract(inputs, args.blob,oversample=True) # take the only center crop of the image. Intended to change later on
            
            for e in range(batch_extraction.shape[0]):
                        batch_extraction[e]=batch_extraction[e]/LA.norm(batch_extraction[e])
                        datum = caffe.proto.caffe_pb2.Datum()
                        datum.channels = batch_extraction.shape[1]
                        datum.height = 1
                        datum.width = 1
                        datum.data = batch_extraction[e].tostring()
                        #example id
                        id=num_batch*batch_size+e
                        datum.label=int(labels[id])
                        str_id = '{:08}'.format(id)
                        print(str_id)
                        txn.put(str_id,datum.SerializeToString())
                        
#         if num_batch==0:
#             extractions=batch_extraction
#         else:
#             extractions=np.concatenate((extractions,batch_extraction),axis=0)
        #end last batch
    
    
#    print("Our final predictions.shape is", extractions.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("extracted features for %d inputs." % len(inputs))

    # Classify.
#     predictions = classifier.predict(inputs, not args.center_only)
    print("Done in %.2f s." % (time.time() - start))
# 
#     # Save
#     print("Saving results into %s" % args.output_file)
#     np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)
