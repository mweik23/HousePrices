from pathlib import Path
#-----For use in .py files-----#
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#------------------------------#
import sys
sys.path.append(str(PROJECT_ROOT / "src"))
from HousePrices.utils import io
from argparse import ArgumentParser

CONFIG_PATH = str(PROJECT_ROOT / 'config')

def build_parser():
    parser = ArgumentParser(description="Generate model configuration.")
    parser.add_argument('--config_path', type=str, default=CONFIG_PATH, help='Path to configuration directory')
    parser.add_argument('--output_name', type=str, default='model_config.json', help='Output name for the model config JSON file')
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    model_info = {
        "model_name": "catboostregressor",
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3,
        "bagging_temperature": 1.0,
        "random_seed": 42,
        "task_type": "CPU",
        "od_type": "Iter",
        "od_wait": 50
    }
    # Ensure the output directory exists
    output_path = Path(args.config_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # check output name is json
    if not args.output_name.endswith('.json'):
        raise ValueError("Output name must end with .json")

    # Save the model configuration to a JSON file
    io.save_json(model_info, f'{args.config_path}/{args.output_name}')

if __name__ == "__main__":
    main()
    
    
