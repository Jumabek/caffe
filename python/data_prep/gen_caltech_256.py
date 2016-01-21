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
    print("directory",pycaffe_dir)
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "dataset_root",
        help="the directory whole dataset contains, i.e data/caltech_256"
    )
    parser.add_argument(
        "--list",
        action='store_true',
        help="Switch for list generation mode ."
    )
    args = parser.parse_args()
    
    # class looks like "data/caltech_256/054.diamond-ring"
    classes=[x[0] for x in os.walk(join(args.dataset_root,"images")) if not isfile(join(join(args.dataset_root,"images"), x[0])) ]
    
    #classes[0] is caltech_256 itself
    classes=classes[1:]
    
    
    # labels looks like "data/caltech_256/054"
    labels= [x.split(".")[0] for x in classes]
     
    # labels looks like "054"
 
    labels= [int(x.split("/")[-1]) for x in labels]
    
     

    sorted_indices=[i[0] for i in sorted(enumerate(labels),key=lambda x:x[1])]    
    
    #sort the labels in order
    sorted_classes=[]
    sorted_labels=[]
    for i in range(len(classes)):
        sorted_classes.append(classes[sorted_indices[i]])
        sorted_labels.append(labels[sorted_indices[i]])
        #print(sorted_classes[i])
    
    #from now on we will work only with 'sorted_classes' so lets make 'classes' equal to 'sorted_classes'
    classes=sorted_classes
    labels=sorted_labels
    
    
    
    train_stream_list=open(join(args.dataset_root, "train_list.txt"),"w")
    test_stream_list=open(join(args.dataset_root, "test_list.txt"),"w")
    train_stream=open(join(args.dataset_root, "train.txt"),"w")
    test_stream=open(join(args.dataset_root, "test.txt"),"w")
    
    for c in range(len(sorted_classes)):
        examples=[f for f in listdir(sorted_classes[c]) if isfile(join(sorted_classes[c],f))]
        #print(len(examples))
        #in order to make it easy to permutate
        examples=np.array(examples)

        examples=np.random.permutation(examples)
        
        #write input for for general purpose
        for example in examples[:60]:
            train_stream.write(join(sorted_classes[c],example))
            train_stream.write("\t")
            train_stream.write(str(labels[c]))
            train_stream.write("\n")
        
        for example in examples[60:]:
            test_stream.write(join(sorted_classes[c],example))
            test_stream.write("\t")
            test_stream.write(str(labels[c]))
            test_stream.write("\n")
        
        #write input for caffe featue extraction
        for example in examples[:60]:
            train_stream_list.write(join(sorted_classes[c],example))
            train_stream_list.write("\n")
        
        for example in examples[60:]:
            test_stream_list.write(join(sorted_classes[c],example))
            test_stream_list.write("\n")
        
        
if __name__ == '__main__':
    main(sys.argv)



