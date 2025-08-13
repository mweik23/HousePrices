import os
import re
from tabnanny import verbose
import pandas as pd
import json
from collections import Counter
import numpy as np

def setup_expdir(expdir, overwrite=False):
    if not expdir.exists():
        print(f"Creating experiment directory: {expdir}")
        expdir.mkdir(parents=True, exist_ok=True)
    else:
        if overwrite:
            print(f"Experiment directory {expdir} already exists. Overwriting.")
            os.system(f'rm -r {expdir}')
            expdir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Experiment directory {expdir} already exists. Not overwriting.")
            raise FileExistsError(f"Experiment directory {expdir} already exists. Set 'overwrite' to True to overwrite.")

#define paths and names
def get_paths(project_root, datadir=None):
    if datadir is None:
        datadir = project_root / 'data'
    config_dir = project_root / 'config'
    return datadir, config_dir

#TODO: clean up directory specification
def get_data(data_path, load_info=True, print_output=False, print_status=True, split='all'):
    if print_status:
        print('loading data and info...')
    
    #load raw data and info
    if load_info:
        with open(data_path / 'raw/data_description.txt', 'r') as file:
            data_info = file.read()
    else:
        data_info = None

    # if split is 'all', load all data. Train val split will be done on the outside
    if split == 'all':
        data_in = pd.read_csv(data_path / 'raw/train.csv', keep_default_na=False, na_values=['_'])
    elif split == 'test':
        data_in = pd.read_csv(data_path, keep_default_na=False, na_values=['_'])
    else:
        data_in = pd.read_csv(data_path /  f'tvt/{split}.csv', keep_default_na=False, na_values=['_'])

    if print_output:
        if data_info is not None:
            print('data info: ')
            print(data_info)
        print('raw data: ')
        print(data_in)
    
    if 'Id' in data_in.columns:
        Ids = data_in.pop('Id')
    else:
        Ids = None
    return Ids, data_in, data_info

def parse_info(data_info, print_output=False, print_status=True):
    if print_status:
        print('parsing data info file for data options...')
    x = re.findall('^\w+:.+?\n\s*\n', data_info, flags=re.MULTILINE)
    categs = [heading.split(':')[0] for heading in x]
    descripts = re.split('^\w+:.+?\n\s*\n', data_info, flags=re.MULTILINE)[1:]
    options = [re.findall('^\s+([\w|\.|\&| |\(\|\)]+)\t', des, flags=re.MULTILINE) for des in descripts]
    options_dict = {cat:[o for o in opt if o.strip()] for cat, opt in zip(categs, options)}
    if print_output:
        print('number of descriptions: ', len(descripts))
        print('numbers of categories: ', len(categs))
        print('options for each category')
        for k, v in options_dict.items():
            print(f"{k}: {v}")
    raw_numeric = []
    categorical = []
    for k, v in options_dict.items():
        if len(v)==0:
            raw_numeric.append(k)
        else:   
            categorical.append(k)
    return options_dict, raw_numeric, categorical

def check_numeric_columns(X, raw_numeric, scalar_inputs, verbose=False):
    """Check if numeric columns contain non-numeric data."""
    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False
    non_numeric_list = []
    X = X.copy()
    #check raw number columns to see if they contain non-numerical data
    for k in raw_numeric:
        numeric = X[k].apply(is_number)
        if np.any(~numeric):
            print(f"Warning: Column {k} contains non-numeric data:")
            non_numeric = X[k][~numeric].value_counts()
            if verbose:
                print(non_numeric)
            non_numeric_list.append(non_numeric)
    if len(non_numeric_list) == 0:
        print("All raw numeric columns contain only numeric data.")
    #check scalar inputs to see if they are in raw numeric
    n=0
    for input in scalar_inputs:
        if input not in raw_numeric:
            n += 1
            print(f"Warning: {input} is in scalar_inputs but not in raw_numeric. It will not be processed as a numeric input.")
    if n == 0:
        print("All scalar_inputs are present in raw_numeric.")
    return non_numeric_list

def update_categories(options_dict, categorical, X, ignore_thresh=20):
    n=0
    m=0
    data_in = X.copy()
    for k in categorical:
        if k in data_in.columns:
            #convert to string if it is categorical
            if type(data_in[k][0]) != str:
                data_in[k] = data_in[k].apply(str)
            #check for unexpected values
            actual_opts = list(set(data_in[k].tolist()))
            if not all([item in options_dict[k] for item in actual_opts]):
                n+=1
                missing_opts = [item for item in actual_opts if item not in options_dict[k]]
                #check frequency of missing options
                freq = Counter(data_in[k])
                for item in missing_opts:
                    item_freq = freq[item]
                    if item_freq >= ignore_thresh:
                        options_dict[k].append(item)
                        print(f"Adding option {item} to options_dict for {k} with frequency {item_freq}")
                    else:
                        print(f"Ignoring option {item} for {k} with frequency {item_freq} below threshold {ignore_thresh}")
            
                print(f"Updated options_dict for {k}: {options_dict[k]}")
        else:
            m+=1
            print(f'key {k} is not a column in data_in')
    if n == 0:
        print("All categorical keys are present in options_dict.")
    if m == 0:
        print("All categorical keys are present in data_in columns.")
    return options_dict

def load_json(file_path, print_output=False):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        if print_output:
            print(f"Loaded JSON from {file_path}:")
            print(json.dumps(data, indent=2))
    return data

def save_json(data, file_path):
    """Save a dictionary as a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {file_path}")
