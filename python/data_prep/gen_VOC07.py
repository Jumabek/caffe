'''
Created on Jan 21, 2016

@author: jumabek
'''

import os
import sys
import argparse
from os.path import join,isfile 
from os import listdir
from random import random
import numpy as np

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    print(" pycaffe_dir",pycaffe_dir)
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "dataset_root",
        help="the directory whole dataset contains, i.e data/VOC07_train or data/VOC07_test"
    )
    parser.add_argument(
        "mode",
        help="i.e. train/test"
    )
    args = parser.parse_args()
    
    classes=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    
    train_stream=open(join(args.dataset_root,args.mode + ".txt"),"w")
    
    written_names=[]
    for i in range(len(classes)):
        if args.mode=="train":
            f=open(join(args.dataset_root,"ImageSets","Main",classes[i]+"_trainval.txt"))
        else:
            f=open(join(args.dataset_root,"ImageSets","Main",classes[i]+"_test.txt"))
        lines=[line.rstrip("\n") for line in f.readlines()]
        
        names=[line.split()[0] for line in lines]
        labels=[line.split()[1] for line in lines]
        
        for l in range(len(labels)):
            if labels[l]=="1" and names[l] not in written_names:
                train_stream.write("%s\t%d\n"%(join("/home/jumabek/research/caffe/data/VOC07/images",names[l]+".jpg"),i+1))
                written_names.append(names[l])
                
        
    
    
    
if __name__ == '__main__':
    main(sys.argv)



