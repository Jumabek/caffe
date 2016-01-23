'''
Created on Jan 22, 2016

@author: jumabek
'''

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
        help="path/to/fc6/train,path/to/fc7/train as lmdb"
    )
    parser.add_argument(
        "output_file",
        help="Output arff filename. i.e. path/to/MixNet/fc6_fc7/train.svmlight"
    )   
    args = parser.parse_args()

    write_stream=open(args.output_file,"w")
    args.input_file=args.input_file.split(",")
    num_layers=len(args.input_file)
    txns=[]
    for l in range(num_layers):
        env=lmdb.open(args.input_file[l],readonly=True)
        txn=env.begin()
        txns.append(txn)
    
    #now start writing the data
    cursor=txns[0].cursor()
    #print("after cursor")
    for key, value in cursor:
        so_far_len=1
        for l in range(num_layers):
            raw_datum = txns[l].get(key)
            datum=caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.float32)            
            
            #write the label for svmlight format
            if l==0:
                y=datum.label
                write_stream.write("%d\t"%y)
                
            for i in range(len(flat_x)):
                write_stream.write("%d:%f "%((so_far_len+i), flat_x[i]))
            so_far_len+=len(flat_x)
        write_stream.write("\n")
        
        
            
            
            


























if __name__ == '__main__':
    main(sys.argv)
