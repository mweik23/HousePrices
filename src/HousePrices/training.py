import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
import os

#-----For use in .py files-----#
project_root = Path(__file__).resolve().parents[2]
#------------------------------#

sys.path.append(str(project_root / "src"))
from HousePrices.utils import io
from HousePrices.utils.models import ModelConfig

def prepare_data(data_paths, target_name='', pre_split=False, val_frac=0.2, tv_split_seed=42):
    if pre_split:
        print("Using pre-split data. Train, val split with be ignored")
        # load training data
        data_train, data_info = io.get_data(data_paths, split='train')
        X_train, y_train = data_train.drop(columns=[target_name]), data_train[target_name]

        # load validation data
        data_val, _ = io.get_data(data_paths, split='val')
        X_val, y_val = data_val.drop(columns=[target_name]), data_val[target_name]
    else:
        # Load the entire dataset
        data_in, data_info = io.get_data(data_paths, split='all')
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            data_in.drop(columns=[target_name]),
            data_in[target_name],
            test_size=val_frac,
            random_state=tv_split_seed
        )
    # Parse the data info to get features info for preprocessing
    features_dict, raw_numeric, categorical = io.parse_info(data_info, print_output=False)

    # load transformations config file
    transformations_path = data_paths['config'] + '/transformations.json'
    transformations = io.load_json(transformations_path, print_output=False)

    # modify transformations to include more feature information
    non_ordinal = [feat for feat in categorical if feat not in transformations['ordinal']] # categorical features that are not ordinal
    transformations['non_ordinal'] = non_ordinal
    transformations['features_dict'] = features_dict
    transformations['raw_numeric'] = raw_numeric
    transformations['categorical'] = categorical

    # add additional categories to features_dict
    features_dict = io.update_categories(features_dict, non_ordinal, pd.concat([X_train, X_val], ignore_index=True))

    # Check for non-numeric data in raw numeric columns
    _ = io.check_numeric_columns(data_in, raw_numeric, transformations['scalar_inputs'])
    return X_train, y_train, X_val, y_val, transformations, features_dict

def get_model_config(model_params, model_config_path, expdir, model_name):
    if model_params is None:
        if os.path.exists(model_config_path):
            model_params = io.load_json(model_config_path)
        else:
            raise FileNotFoundError(f"Model configuration file {model_config_path} not found and no model parameters provided.")
    else:
        # Create model configuration from command line arguments
        model_params = model_params
        #save model parameters to file
        io.save_json(model_params, model_config_path)
    # TODO: may need to make this more robust
    if model_name == 'catboost_regressor':
        model_params['train_dir'] = str(expdir)  # Ensure train_dir is set
    else:
        print(f"Warning: No special handling for model {model_name}. Save directory not set.")

    # build model
    model_config = ModelConfig(
        name=model_name,
        params=model_params
    )
    return model_config

def evaluate_model(pipe, X_val, y_val):
    # Check metrics on validation set
    print("Evaluating the model...")
    val_preds = pipe.predict(X_val)
    rmse = root_mean_squared_error(y_val, val_preds)
    print(f"Validation RMSE: {rmse:.4f}")
    return val_preds, rmse