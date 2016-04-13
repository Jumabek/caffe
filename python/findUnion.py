import numpy as np
import os
import sys
import argparse
import glob
import time
import lmdb

import caffe
from skimage.viewer.canvastools.painttool import LABELS_CMAP

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="path/to/fc6/predictions.svm,path/to/fc7/prediction.svm "
    )
    parser.add_argument(
        "gt_file",
        help="test.txt"
    )

    parser.add_argument(
        "output_file",
        help="Output filename. i.e. path/to/union.svm"
    )   
    args = parser.parse_args()
    args.input_file=args.input_file.split(",")

    input_stream=[]
    layers=[]
    for j in range(len(args.input_file)):
        input_stream.append(open(args.input_file[j]))
        layer=[line.rstrip("\n") for line in open(args.input_file[j]).readlines() ]
        layers.append(layer)
    
    gt_stream = open(args.gt_file);
    test=[line.rstrip("\n").split()[1] for line in gt_stream.readlines()]

    output_stream = open(args.output_file,"w");
     
    
    l = 0
    min_len=len(layers[0])
    for j in range(len(layers)):
        if len(layers[j])<min_len:
            min_len=len(layers[j])
    
    
    for i in range(min_len):
        correct=False
        for j in range(len(layers)):
            print(layers[j][i],test[i])
            if layers[j][i]==test[i]:
                correct=True
        if correct:
            l =l+1
            output_stream.write('1\n')
        else:
            output_stream.write("0\n")
    
   
    output_stream.write("%s %%"%str(float(l)/float(min_len)))
        
if __name__ == '__main__':
    main(sys.argv)





