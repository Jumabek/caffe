'''
Created on Jan 31, 2017

@author: jumabek
'''


import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', metavar='filenamelist', type=str, default='trainlist.txt',
                    help='filename containing image filenames in caffe label format')
parser.add_argument('-o', metavar='npy_filenames', type=str, default='filenames.npy',
                    help='npy filename to save filenames')
parser.add_argument('-g', metavar='gpu', type=int, default=-1,
                    help='GPU device ID (CPU if this negative)')
args = parser.parse_args()

f = open(args.i)

filenames = [filename.rstrip("\r\n").split()[0] for filename in f.readlines()]


filenames = np.array(filenames)

np.save(args.o,filenames)

