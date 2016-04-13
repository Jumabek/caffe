'''
Created on Mar 18, 2016

@author: jumabek
'''

import os
import sys
import argparse
from os.path import join,isfile 
from os import listdir
from random import random
import numpy as np

PREFIX_PATH="/home/jumabek/research/caffe/data/fcd"

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    print(" pycaffe_dir",pycaffe_dir)
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "dataset_root",
        help="the directory whole dataset contains, i.e data/images"
    )
    args = parser.parse_args()
    
    train_stream=open(join(args.dataset_root,"train.txt"),"w")
    test_stream=open(join(args.dataset_root,"test.txt"),"w")
    
    
    #train set    
    class_dirs=[x[0] for x in os.walk(join(args.dataset_root,"images","train")) ]
    class_dirs=class_dirs[1:]
    
    class_dirs=sorted(class_dirs)
    
    for i in range(len(class_dirs)):
        files=[x for x in listdir(class_dirs[i]) if isfile(join(class_dirs[i],x))]
        index=class_dirs[i].find("images")
        path2image= class_dirs[i][index:]
        for file in files:
            
            train_stream.write("%s\t%d\n"%(join(PREFIX_PATH, path2image,file),i))
    

    #test set
    class_dirs=[x[0] for x in os.walk(join(args.dataset_root,"images","test")) ]
    class_dirs=class_dirs[1:]
    
    class_dirs=sorted(class_dirs)
    
    for i in range(len(class_dirs)):
        files=[x for x in listdir(class_dirs[i]) if isfile(join(class_dirs[i],x))]
        index=class_dirs[i].find("images")
        path2image= class_dirs[i][index:]
        for file in files:
            
            test_stream.write("%s\t%d\n"%(join(PREFIX_PATH,path2image,file),i))
    
    
    
if __name__ == '__main__':
    main(sys.argv)



