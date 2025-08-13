import joblib
import pandas as pd
import shutil
from pathlib import Path
import time
import sys
from argparse import ArgumentParser
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from HousePrices.utils import io


def build_parser():
    parser = ArgumentParser(description="Train a regression model on House Prices dataset.")
    parser.add_argument('--model_path', type=str, default="logs/my_experiment/pipeline.joblib", help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default="data/raw/test.csv", help='Path to the new data for prediction')
    #TODO: update default output directory
    parser.add_argument('--output_dir', type=str, default="predictions/run", help='Directory to save predictions')
    parser.add_argument('--base_dir', type=str, default=str(PROJECT_ROOT), help='Base directory of the project')
    parser.add_argument('--train_name', type=str, default='', help='Name of the experiment to load the model from')
    parser.add_argument('--out_var', type=str, default='SalePrice', help='Name of the output variable in the predictions file')
    return parser

def main(argv=None):
    #get date and time for experiment
    run_time = time.strftime("%Y%m%d-%H%M%S")

    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)

    #construct the model path
    if args.train_name != '':
        # If train_name is provided, construct the model path from it
        model_path = PROJECT_ROOT / 'logs' / args.train_name / 'pipeline.joblib'
    else:
        # Otherwise, use the provided model path
        model_path = Path(args.model_path)
    
    #create output directory if it doesn't exist
    output_dir = Path(PROJECT_ROOT / f'{args.output_dir}__{run_time}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = joblib.load(model_path)

    #load test data
    full_data_path = PROJECT_ROOT / args.data_path
    Ids, X_test, _ = io.get_data(full_data_path, split='test', load_info=False, print_output=False)
 
    # Predict
    preds = pipe.predict(X_test)
    df_preds = pd.DataFrame({
        Ids.name: Ids,
        args.out_var: preds
    })

    # Save predictions
    out_path = output_dir / f"predictions.csv"
    df_preds.to_csv(out_path, index=False)

    #save arguments to output directory
    io.save_json(vars(args), output_dir / 'test_config.json')

    #copy training config to output directory if it exists
    training_config = model_path.parent / "experiment_config.json"
    if training_config.exists():
        shutil.copy(training_config, output_dir / "experiment_config.json")
    else:
        print(f"Warning: No config found at {training_config}")

    print(f"Predictions and metadata saved to {out_path}")

if __name__ == "__main__":
    main()