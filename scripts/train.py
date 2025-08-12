
# scripts/train.py
from collections import Counter
import sys
from sklearn.pipeline import make_pipeline
from pathlib import Path
from argparse import ArgumentParser

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

def build_parser():
    parser = ArgumentParser(description="Train a regression model on House Prices dataset.")
    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment')
    parser.add_argument('--pre_split', action='store_true', help='Use pre-split data')
    parser.add_argument('--val_frac', type=float, default=0.3, help='Fraction of data to use for validation')
    parser.add_argument('--tv_split_seed', type=int, default=99, help='Random seed for train/validation split')
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOGDIR, help='Directory to save logs and models')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing experiment directory')
    parser.add_argument('--model_name', type=str, default='catboost_regressor', help='Name of the model to train')
    parser.add_argument('--model_params', type=str, default='{}', help='JSON string of model parameters')
    parser.add_argument('--model_config', type=str, default='model_config.json', help='Path to model configuration file (optional)')
    parser.add_argument('--config_path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to configuration directory')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATADIR, help='Path to data directory')
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Create experiment directory
    expdir = f'{args.logdir}/{args.exp_name}'
    # Setup experiment directory
    io.setup_expdir(expdir, overwrite=args.overwrite)

    # Define data paths
    data_paths = io.get_paths(PROJECT_ROOT, datadir=args.data_path)

    # Prepare the data
    X_train, y_train, X_val, y_val, transformations, _ = prepare_data(data_paths,
                                                                      target_name='SalePrice',
                                                                      pre_split=args.pre_split,
                                                                      val_frac=args.val_frac,
                                                                      tv_split_seed=args.tv_split_seed)

    # Build the preprocessor
    preproc = build_preprocessor(transformations)

    # get model configuration
    model_config_path = args.config_path + '/' + args.model_config
    model_config = get_model_config(args.model_params, model_config_path, expdir, args.model_name)

    # build the model
    model = build_model(model_config)
    
    # Create a pipeline with the preprocessor and model
    pipe = make_pipeline(preproc, model)

    # Fit the pipeline on the training data
    print("Fitting the model...")
    pipe.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=100
    )
    # get best iteration from model
    best_iteration = model.get_best_iteration()
    print(f"Best iteration: {best_iteration}")

    val_preds, rmse = evaluate_model(pipe, X_val, y_val)
    print(f"Validation RMSE: {rmse}")
    val_res = {
        "rmse": rmse,
        "best_iteration": best_iteration
    }
    # Save validation results
    io.save_json(val_res, f'{expdir}/val_results.json')
if __name__ == "__main__":
    main()