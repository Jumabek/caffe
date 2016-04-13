'''
Created on Apr 5, 2016

@author: jumabek
'''

import numpy as np
import os
import sys
import argparse
import glob
import time
import lmdb
import copy

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
    parser.add_argument(
        "selection_file",
        help="selected feature indices filename. i.e. path/to/MixNet/fc6_fc7/selectedFeatures.txt"
    )   
    parser.add_argument(
        "num_dims",
        help="How many dimensions you want feature space to be"
    )
    args = parser.parse_args()

    write_stream=open(args.output_file,"w")
    args.input_file=args.input_file.split(",")
    
     
      
    num_layers=len(args.input_file)
    
    feature_stream=open(args.selection_file)
    selected_indices= [int(line.rstrip("\n")) for line in feature_stream.readlines() ]
    selected_indices=selected_indices[:int(args.num_dims)]
    selected_indices=sorted(selected_indices)
    
    txns=[]
    for l in range(num_layers):
        env=lmdb.open(args.input_file[l],readonly=True)
        txn=env.begin()
        txns.append(txn)
    
    #now start writing the data
    cursor=txns[0].cursor()
    #print("after cursor")
    for key, value in cursor:
      
        for l in range(num_layers):
            raw_datum = txns[l].get(key)
            datum=caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.float32)            
            
            if l==0:
                combined_x=flat_x
            else:
                combined_x=np.concatenate((combined_x,flat_x),axis=0)

        #write the label for svmlight format
        y=datum.label
        write_stream.write("%d\t"%y)
            
        #for i in range(len(flat_x)):
        for index in selected_indices:
            write_stream.write("%d:%f "%(index, combined_x[index]))
        
        write_stream.write("\n")      
        
            
            
            

def slice(list,begin,end):
    #print(list)
    
    copylist=copy.deepcopy(list)
    left=find_begin(copylist,begin,len(copylist)/2)
    
    
    copylist=copy.deepcopy(list)
    right=find_end(copylist,end,len(copylist)/2)
   
    
    print("-------------------------list[left]-------------------------------")
    print(int(left))
    print(list[left])
    
    print("-------------------------list[right]-------------------------------")
    print(int(right))
    print(list[right])
    
    return list[left:right+1]

def find_begin(list,begin,or_index):
    pivot=len(list)/2
    print(len(list),pivot,begin,or_index)
    
    if  pivot==0: 
        return or_index
    else:
        if list[pivot]<=begin:
            or_index=or_index+(pivot)/2
            return find_begin(list[pivot:],begin, or_index)
        else:
            or_index=or_index-(pivot-(pivot)/2)
            return find_begin(list[:pivot], begin,or_index)


def find_end(list,end,or_index):
    pivot=len(list)/2
    print(len(list),pivot,end,or_index)
    
    if  pivot==0: 
        return or_index
    else:
        if list[pivot]<end:
            or_index=or_index+(pivot)/2
            return find_end(list[pivot:],end, or_index)
        else:
            or_index=or_index-(pivot-(pivot)/2)
            return find_end(list[:pivot], end,or_index)




















if __name__ == '__main__':
    main(sys.argv)
