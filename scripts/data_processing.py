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

def get_data(data_paths, print_output=False, split='train'):
    #create directory for processed data
    if not os.path.isdir(data_paths['processed']):
        os.system('mkdir ' + data_paths['processed'])
    if split=='train':
        #load raw data and info
        with open(data_paths['raw'] + '/data_description.txt', 'r') as file:
            data_info = file.read()
    else:
        data_info=None
    ## this will need to be done for train test and val
    data_in = pd.read_csv(data_paths['tvt'] + '/' + split + '.csv', keep_default_na=False, na_values=['_'])
    if print_output:
        if data_info is not None:
            print('data info: ')
            print(data_info)
        print('raw data: ')
        print(data_in)
    return data_in, data_info

def parse_info(data_info, print_output=False):
    x = re.findall('^\w+:.+?\n\s*\n', data_info, flags=re.MULTILINE)
    categs = [heading.split(':')[0] for heading in x]
    descripts = re.split('^\w+:.+?\n\s*\n', data_info, flags=re.MULTILINE)[1:]
    options = [re.findall('^\s+([\w|\.|\&| |\(\|\)]+)\t', des, flags=re.MULTILINE) for des in descripts]
    options_dict = {cat:opt for cat, opt in zip(categs, options)}
    if print_output:
        print('number of descriptions: ', len(descripts))
        print('numbers of categories: ', len(categs))
        print('options for each category')
        for k, v in options_dict.items():
            print(f"{k}: {v}")
    return options_dict

def change_types(data_in, options_dict, trans, print_output=False):
    #get types in options_dict to match the types in data_in
    for k in data_in.keys():
        if k in options_dict:
            if k in trans['one_hot']+list(trans['one_hot_plus'].keys()):
                if type(data_in[k][0])!=str:
                    data_in[k] = data_in[k].apply(str)
            #gather the unique elements in data_in[k], the set of all values of the variable with heading k
            actual_opts = list(set(data_in[k].tolist()))
            actual_type = type(actual_opts[0])
            #if the type of the variable is something other than str, need to convert options_dict variable options to the correct type
            if options_dict[k] == []:  #this represents a free numeric entry
                options_dict[k] = {'variable_type': actual_type}  #what type is this entry in data_in to start?
                #keep working here
            else:
                #if the type of a column in data_in is something other than a string, 
                #the options of the corresponding key in options_dict will have to be converted 
                if actual_type != str: 
                    options_dict[k] = [actual_type(item) for item in options_dict[k]]

                #check if there exist any items in data_in that do not appear in opitons_dict and raise an error if so
                check_options = [item in options_dict[k] for item in actual_opts]
                if not all(check_options):
                    #print(check_options)
                    false_idx = -1
                    num_false = check_options.count(False)
                    for n in range(num_false):
                        false_idx = check_options.index(False, false_idx+1)
                        print(f'ERROR: value {actual_opts[false_idx]} for key {k} does not exist in options_dict')           
        #if this key does not exist in options_dict and it is not a special key, then raise an error
        elif k !='SalePrice' and k!= 'Id':
            print(f'ERROR: key {k} in the data is not in options_dict')
    if print_output:
        print('Intermediate result for options_dict:')
        for k, v in options_dict.items():
            print(f"{k}: {v}")
            if type(v)==list:
                print('type of options: ', type(v[0]))
            print('type of actual: ', type(data_in[k].loc[0]))

    float_type = float
    for k, v in data_in.items():
        all_numeric=False
        free_var=False
        if k in options_dict:
            #print(k)
            #check if all options are numeric 
            if type(options_dict[k])==list: #if options are explicitly listed and they are strings then check if they are strings of numeric values
                if type(options_dict[k][0])==str:
                    numeric = [item.replace('.', '').replace('-', '').replace(' ', '').isnumeric() for item in options_dict[k]] #remove special characters
                    all_numeric = all(numeric)
            elif type(options_dict[k])==dict: #if options are described by a dictionary then the the data type is a free form number
                if options_dict[k]['variable_type']==str: #if that number is currently in the form of a string then it must be converted
                    all_numeric=True
                    free_var=True
            if all_numeric and k != 'MSSubClass':
                #print(options_dict[k])
                #before converting, replace 'NA' with -1
                data_in[k] = data_in[k].replace('NA', -1)
                data_in[k] = data_in[k].astype(float_type) #convert data
                #print('type: ', type(data_in[k].loc[1]))

                #convert options dict as well
                if free_var:
                    options_dict[k]['variable_type']=float_type
                else:
                    options_dict[k] = [float_type(item) for item in options_dict[k]]
    if print_output:
        print('Final result for options_dict:')
        for k, v in options_dict.items():
            print(f"{k}: {v}")
            if type(v)==list:
                print('type of options: ', type(v[0]))
            print('type of actual: ', type(data_in[k].loc[0]))

    return data_in, options_dict

def add_to_series(old_series, pos, new_label, val):
    new_index  = old_series.index.insert(pos, new_label)
    # insert into the underlying array of values
    new_values = np.insert(old_series.values, pos, val)
    new_series = pd.Series(new_values, index=new_index)
    return new_series

def input_dist(df_new, key, dist=None):
    if dist is None:
        dist = df_new[df_new[key]!=1.].mean() #find distribution of different values excluding unknowns
    else:
        dist = add_to_series(dist, df_new.columns.get_loc(key), key, 0) #add an entry to dist for key
    df_new[df_new[key]==1] = dist #Give the unknowns the distribution of inputs
    #drop unknown column from df_new and dist
    df_new = df_new.drop(columns=[key])
    dist = dist.drop(index=key)
    return df_new, dist

def to_one_hot(data_in, categ, options, dtype=float):
    num_data = len(data_in)
    one_col = data_in[categ] #extract the column we want to transform
    actual_opts = list(set(one_col.tolist())) #extract possible values entries in that column have
    #print(actual_opts)
    to_add = []
    for item in options:
        #add missing entries
        if item not in actual_opts:
            one_col.loc[len(one_col)] = item  # adding a row
    df1 = pd.get_dummies(one_col, dtype=dtype) #transform to one hot encoding
    df1 = df1.loc[:num_data-1] #remove extra rows
    return df1

def to_one_hot_plus(data_in, categ, options, conv_dict, dist=None):
    df_new = to_one_hot(data_in, categ, options) #transform to one hot encoding
    #consider special cases
    for k2, val in conv_dict.items():
        if val==0:
            df_new = df_new.drop(columns=[k2])
        elif val=='dist':
            df_new, dist = input_dist(df_new, k2, dist=dist)
        elif k2=='weight':
            df_new = df_new.mul(data_in[val], axis=0)
    if dist is not None:
        dist = dist[dist.index.isin(df_new.columns)]

    return df_new, dist

def create_mappings(to_scalar, options_dict):
    mappings = {}
    for k in to_scalar:
        num_opts = len(options_dict[k])
        
        # define mapping in descending order from 1 to 0.
        mappings[k] = {opt: 1-i/(num_opts-1) for i, opt in enumerate(options_dict[k])}
    return mappings

def cat_to_vec(init_df, mapping, new_cols=None, cat=None):
    if cat is None:
        cat = init_df.columns[0]
    vec = init_df[cat].map(mapping)

    # turn list‐values into real columns
    vec_df = pd.DataFrame(vec.tolist(),
                    index=init_df.index,
                    columns=new_cols)
    return vec_df

def level_transform(data_in, options_dict, cols, print_output=False):
    #transform columns in the to_scalar category
    mappings = create_mappings(cols, options_dict)
    for k in cols:
        data_in[k] = data_in[k].map(mappings[k]) #map strings to numbers
    data_in = data_in.drop(columns='TotalBsmtSF') #remove TotalBsmtSF becasue it is redundant
    data_in['YrSold'] = data_in['YrSold'] + data_in['MoSold']/12 - 1/24 # convert year and month into a single entry for year
    data_in = data_in.drop(columns='MoSold') #drop month column
    if print_output:
        print(data_in.columns)
    return data_in

def convert_to_str(data_in, cols):
    for key in cols:
        if type(data_in[key][0])!=str:
            print('converting column ' + key + ' to a str.')
            data_in[key] = data_in[key].apply(str)
    return data_in

def create_new_cols(data_in, options_dict, transformations, dists_in=None):
    #initialize dfs and dists_out
    dfs={}
    dists_out = {}
    #convert dists_in to a dictionary if it isn't one
    if dists_in is None:
        dists_in = {}

    # transform those in the one_hot_plus category
    for k, val in transformations['one_hot_plus'].items():
        if k in dists_in.keys():
            dist = dists_in[k] # if there is a input dist for k then we will use that
        else:
            dist = None # otherwise we will set the dist to None and caluclate 
                        # a new dist if needed (inside to_one_hot_plus())
        df_new, dist = to_one_hot_plus(data_in, k, options_dict[k], val, dist=dist) # do the conversion
        #if a dist was produced then add it to the output set of dists
        if dist is not None:
            dists_out[k] = dist
        dfs[k] = df_new #add new dataframe to df_new

    # transform those in the one_hot category
    for k in transformations['one_hot']:
        dfs[k] = to_one_hot(data_in, k, options_dict[k])

    return dfs, dists_out

   