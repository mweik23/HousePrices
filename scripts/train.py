# scripts/train.py
from collections import Counter
import joblib
import sys
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import time

#-----For use in .py files-----#
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#-----For use in .ipynb files-----#
#PROJECT_ROOT = Path().resolve().parents[0]

sys.path.append(str(PROJECT_ROOT / "src"))
from HousePrices.utils import io
from HousePrices.pipelines.build_preprocessor import build_preprocessor
from HousePrices.utils.models import build_model
from HousePrices.training import prepare_data, evaluate_model, get_model_config

DEFAULT_LOGDIR = str(PROJECT_ROOT / 'logs')
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / 'config')
DEFAULT_DATADIR = str(PROJECT_ROOT / 'data')

'''Example usage:
python scripts/train.py --exp_name my_experiment --val_frac 0.3 --model_config model_config.json
'''

def build_parser():
    parser = ArgumentParser(description="Train a regression model on House Prices dataset.")
    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment')
    parser.add_argument('--pre_split', action='store_true', help='Use pre-split data')
    parser.add_argument('--val_frac', type=float, default=0.3, help='Fraction of data to use for validation')
    parser.add_argument('--tv_split_seed', type=int, default=99, help='Random seed for train/validation split')
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOGDIR, help='Directory to save logs and models')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing experiment directory')
    parser.add_argument('--model_params', type=str, default='{}', help='JSON string of model parameters')
    parser.add_argument('--model_config', type=str, default='model_config.json', help='Path to model configuration file (optional)')
    parser.add_argument('--config_path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to configuration directory')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATADIR, help='Path to data directory')
    return parser

def main(argv=None):
    #get date and time for experiment
    run_time = time.strftime("%Y%m%d-%H%M%S")

    # Build the argument parser
    parser = build_parser()
    args = parser.parse_args(argv)

    # Create experiment directory
    expdir = Path(f'{args.logdir}/{args.exp_name}_{run_time}') if args.exp_name!= '' else Path(f'{args.logdir}/experiment_{run_time}')

    # Setup experiment directory
    io.setup_expdir(expdir, overwrite=args.overwrite)

    #dump args experiment config
    io.save_json(vars(args), expdir / 'experiment_config.json')

    # Define data paths
    data_path, config_path = io.get_paths(PROJECT_ROOT, datadir=Path(args.data_path))

    # Prepare the data
    X_train, y_train, X_val, y_val, transformations, _ = prepare_data(data_path,
                                                                      config_path,
                                                                      target_name='SalePrice',
                                                                      pre_split=args.pre_split,
                                                                      val_frac=args.val_frac,
                                                                      tv_split_seed=args.tv_split_seed)

    # Build the preprocessor
    preproc = build_preprocessor(transformations)

    # get model configuration
    model_config_path = args.config_path + '/' + args.model_config
    model_config = get_model_config(args.model_params, model_config_path, expdir)

    # build the model
    model = build_model(model_config)

    # Ensure train_dir is set
    if hasattr(model, 'set_params'):
        model.set_params(train_dir=str(expdir))  # Ensure train_dir is set for CatBoost

    #Wrap model so it trains on log(y) and predicts on original scale
    log_model = TransformedTargetRegressor(
        regressor=model,
        func=np.log,          # or np.log1p
        inverse_func=np.exp
    )

    # Create a pipeline with the preprocessor and model
    pipe = make_pipeline(preproc, log_model)

    #TODO: wrap this so that fit can be called just once on the pipeline
    #fit preprocessor on training data and transform
    print("Fitting the preprocessor...")
    X_train_transformed = pipe[:-1].fit_transform(X_train, y_train)

    # Transform validation set
    X_val_transformed = pipe[:-1].transform(X_val)
    y_val_transformed = pipe[-1].func(y_val)

    # Fit the pipeline on the training data
    print("Fitting the model...")
    pipe[-1].fit(
        X_train_transformed, y_train,
        eval_set=(X_val_transformed, y_val_transformed),
        use_best_model=True,
        verbose=100
    )

    # get best iteration from model
    best_iteration = pipe[-1].regressor_.get_best_iteration()
    print(f"Best iteration: {best_iteration}")

    #save the trained model
    joblib.dump(pipe, expdir / "pipeline.joblib")

    # also save the underlying CatBoost model separately (more portable)
    pipe[-1].regressor_.save_model(str(expdir / "model.cbm"))

    '''
    to load and test the pipeline later:
    pipe_loaded = joblib.load(expdir / "pipeline.joblib")
    y_pred = pipe_loaded.predict(X_new)  # applies preproc then model
    '''

    # Evaluate the model on the validation set
    val_preds, log_rmse, log_r2 = evaluate_model(pipe, X_val, y_val, y_transform=pipe[-1].func)
    #print(f"Validation RMSE: {log_rmse}")
    val_res = {
        "log_rmse": log_rmse,
        "log_r2": log_r2,
        "best_iteration": best_iteration
    }
    # Save validation results
    io.save_json(val_res, expdir / 'val_results.json')
if __name__ == "__main__":
    main()