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
    args = parser.parse_args()
    
    train_stream=open(join(args.dataset_root,"train.txt"),"w")
    test_stream=open(join(args.dataset_root,"test.txt"),"w")
    
    class_file=join(args.dataset_root,"ClassName.txt")
    f=open(class_file)
    classes=[line.rstrip("\n") for line in f.readlines()]
    for c in range(len(classes)):
        class_dir=join(args.dataset_root,"images"+classes[c])
        print("class_dir",class_dir)
        class_files=[]
        for root, dirs, files in os.walk(class_dir):
                for file in files:
                    class_files.append(file)
        
        
        class_files=np.array(class_files)

        class_files=np.random.permutation(class_files)
        
        for file in class_files[:50]:
            train_stream.write("%s\t%d\n"%(join("/home/jumabek/research/caffe",class_dir,file),c+1))
        for file in class_files[50:100]:
            test_stream.write("%s\t%d\n"%(join("/home/jumabek/research/caffe",class_dir,file),c+1))
        
    
if __name__ == '__main__':
    main(sys.argv)



