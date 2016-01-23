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
        help="Output arff filename. i.e. path/to/MixNet/fc6_fc7/train.arff"
    )   
    parser.add_argument(
        "num_labels",
        help="i.e 257 for caltech-256."
    )
    args = parser.parse_args()

    write_stream=open(args.output_file,"w")
    write_stream.write("@RELATION %s"%args.output_file.split(".")[0])
    write_stream.write("\n\n")
    
    args.input_file=args.input_file.split(",")
    num_layers=len(args.input_file)
    
    feature_dims=0
    #write arff file header
    
    #calc total feature dims
    txns=[]
    for l in range(num_layers):
        env=lmdb.open(args.input_file[l],readonly=True)
        txn=env.begin()
        txns.append(txn)
        raw_datum = txn.get(b'00000000')
        datum=caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.float32)
        feature_dims+=len(flat_x)
                
    write_stream.write("%% %d numeric descriptors\n\n"%feature_dims)
    
    for i in range(1,feature_dims+1):
        write_stream.write("@ATTRIBUTE a%05d NUMERIC\n"%i)
    write_stream.write("@ATTRIBUTE class {")
    for i in range(1,int(args.num_labels)):
        write_stream.write("%d,"%i)
    #write the last class
    write_stream.write("%d"%int(args.num_labels))
    
    write_stream.write("}\n\n\n")
    
    write_stream.write("@DATA\n\n")
        
    #now start writing the data
    cursor=txns[0].cursor()
    #print("after cursor")
    for key, value in cursor:
        for l in range(num_layers):
            raw_datum = txns[l].get(key)
            datum=caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.float32)            
            y=datum.label
            for i in range(len(flat_x)):
                write_stream.write("%f,"%flat_x[i])
        write_stream.write("%d\n"%y)
        
        
            
            
            


























if __name__ == '__main__':
    main(sys.argv)
