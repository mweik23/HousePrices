import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score

#-----For use in .py files-----#
project_root = Path(__file__).resolve().parents[2]
#------------------------------#

sys.path.append(str(project_root / "src"))
from HousePrices.utils import io
from HousePrices.utils.models import ModelConfig

def prepare_data(data_path, config_path, target_name='', pre_split=False, val_frac=0.2, tv_split_seed=42):
    if pre_split:
        print("Using pre-split data. Train, val split with be ignored")
        # load training data
        _, data_train, data_info = io.get_data(data_path, split='train')
        X_train, y_train = data_train.drop(columns=[target_name]), data_train[target_name]

        # load validation data
        _, data_val, _ = io.get_data(data_path, split='val')
        X_val, y_val = data_val.drop(columns=[target_name]), data_val[target_name]
    else:
        # Load the entire dataset
        _, data_in, data_info = io.get_data(data_path, split='all')
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
    transformations_path = config_path / 'transformations.json'
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
    _ = io.check_numeric_columns(X_train, raw_numeric, transformations['scalar_inputs'])
    return X_train, y_train, X_val, y_val, transformations, features_dict

def get_model_config(model_params, model_config_path, expdir):
    if model_params == '{}':
        if os.path.exists(model_config_path):
            model_params = io.load_json(model_config_path)
        else:
            raise FileNotFoundError(f"Model configuration file {model_config_path} not found and no model parameters provided.")
    else:
        #save model parameters to file
        io.save_json(model_params, model_config_path)
    
    if 'model_name' in model_params.keys():
        model_name = model_params.pop('model_name')
    else:
        raise ValueError("Model parameters must contain 'model_name' key.")

    # build model configuration
    model_config = ModelConfig(
        name=model_name,
        params=model_params
    )
    return model_config

def evaluate_model(pipe, X_val, y_val, y_transform=np.identity):
    # Check metrics on validation set
    print("Evaluating the model...")
    val_preds = pipe.predict(X_val)
    rmse = root_mean_squared_error(y_transform(y_val), y_transform(val_preds))
    r2 = r2_score(y_transform(y_val), y_transform(val_preds))
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2: {r2:.4f}")
    return val_preds, rmse, r2