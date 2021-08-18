# explainable_grasp.py
# Written Ian Rankin August 2021
#
# Takes as input grasp points and finds the best explanations for the grasp.

import argparse
import glob
import os.path
from os import path
import numpy as np
import matplotlib.pyplot as plt


def read_shape_points(filename):
    jpg_filename = glob.glob(filename[:-4]+'.jp*g')[0]
    

def read_shape_points(folder):
    #filenames = glob.glob(folder+'*.jp*g')
    filenames = glob.glob(folder+'*.csv')

    print(path.exists('./analyze_metrics.p'))

    print(filenames)




def main():
    parser = argparse.ArgumentParser(description='explainable grasp metrics')
    parser.add_argument('-t', type=str, help='the folder of training csv and images', required=True)
    parser.add_argument('-q', type=str, help='the query csv of shape points', required=True)
    args = parser.parse_args()

    read_shape_points(args.t)



if __name__ == '__main__':
    main()
