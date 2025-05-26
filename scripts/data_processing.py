import os
import pandas as pd
import numpy as np
import re
import sys
from collections import Counter
import itertools

#define paths and names
def get_datapaths(basedir):
    this_path = os.getcwd()
    datadir = this_path.split(basedir)[0] + basedir + '/data'
    datadir_in = datadir + '/tvt'
    datadir_raw = datadir + '/raw'
    processed_name = datadir + '/processed_data'
    return {'tvt': datadir_in, 'raw': datadir_raw, 'processed': processed_name}

def get_data(data_paths, print=False, split='train'):
    #create directory for processed data
    if not os.path.isdir(data_paths['processed']):
        os.system(f'mkdir {data_paths['processed']}')

    #load raw data and info
    with open(datadir_raw + '/data_description.txt', 'r') as file:
        data_info = file.read()
    ## this will need to be done for train test and val
    data_in = pd.read_csv(datadir_in + '/train.csv', keep_default_na=False, na_values=['_'])
    print(data_info)
    print(data_in)