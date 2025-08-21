# House Prices Regressor
An application that predicts house prices based on a set of input features.

## Overview
This project is a FastAPI-based web application that provides an API for predicting house prices. It is containerized with Docker for easy deployment and reproducibility. Disclaimer: this is an educational project. This tool is not going to be very effective at predicting the prices of houses that are on the market now because it is based on a machine learning model that was trained using data from the years 2007-2008.

## Features

- RESTful API powered by FastAPI

- Containerized with Docker for easy deployment

- Machine learning predictions served in real time

- Model type: boosted decision tree trained on data from 2007-2008

- Automatic interactive docs at /docs (Swagger UI)

## Installation
### Prerequisites
- [python 3.9](https://www.python.org/downloads/release/python-390/)
- [Docker](https://www.docker.com/get-started)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Local Development (without Docker)
```bash
# Clone the repository
git clone https://github.com/mweik23/HousePrices.git
cd HousePrices

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```
## Command-Line Usage

### Training the Model
#### Specify Model Configuration
A default model configuration file `model_config.json` is provided. You can edit this file directly or generate a new file using
```bash
python scripts/generate_model_config.py --output_name model_config.json
```
The training algorithm takes the model configuration filename as input. Currently, the training algorithm only supports catboost regressor, but support for other models may be added in the future.

#### Specify Preprocessing Configuration
A default preprocessing configuration file `transformations.json` is provided. You can edit this file directly or generate a new file using
```bash
python scripts/generate_preprocessing_config.py --output_name transformations.json
```
The details of the custom transformations might be a little hard to understand, so feel free to reach out if you have any questions. If in doubt, use the default configuration file. The training algorithm takes the preprocessing configuration filename as input.

#### Run Training Script
```bash
python scripts/train.py --exp_name my_experiment --val_frac 0.3 --model_config model_config.json
```

Training Script Arguments

- **`--exp_name`** (default: `""`)  
  Name of the experiment. Used to label logs and saved models.  

- **`--pre_split`** (default: `False`)  
  If set, use pre-split data from 'tvt' directory inside `--data_path` instead of generating a train/validation split.  

- **`--val_frac`** (default: `0.3`)  
  Fraction of data reserved for validation (e.g., `0.3` = 30%).  

- **`--tv_split_seed`** (default: `99`)  
  Random seed used for train/validation split.  

- **`--logdir`** (default: `DEFAULT_LOGDIR`)  
  Directory where logs and checkpoints are saved.  

- **`--overwrite`** (default: `False`)  
  If set, overwrite the existing experiment directory.  

- **`--model_params`** (default: `'{}'`)  
  JSON string of model hyperparameters (optional). If default is given, model parameters from the model configuration file will be used.  

- **`--model_config`** (default: `"model_config.json"`)  
  Path to a JSON file describing model architecture/config. If `--model_params` is given, a new model configuration file will be generated using the provided parameters and saved this file name at `--config_path`. If `--model_params` is not given, the model configuration will be loaded from this path.

- **`--config_path`** (default: `DEFAULT_CONFIG_PATH`)  
  Path to the configuration directory.  

- **`--data_path`** (default: `DEFAULT_DATADIR`)  
  Path to the main data directory.  

- **`--preproc_config_name`** (default: `"transformations.json"`)  
  Name of a JSON file that defines preprocessing/transformations located in `--config_path`. 

Anytime a new training experiment is run, a new directory will be created in `--logdir` with the name `{exp_name}__{timestamp}`. This directory will contain the model checkpoints and logs for that experiment.

### Predicting with the Model
```bash
python scripts/predict.py --train_name my_experiment
```

Prediction Script Arguments

- **`--model_path`** (default: `"logs/my_experiment/pipeline.joblib"`)  
  Path to the trained model file (`.joblib`). If `--train_name` is provided, this can be inferred automatically.  

- **`--data_path`** (default: `"data/"`)  
  Path to the new data on which predictions will be made.  

- **`--output_dir`** (default: `"predictions/run"`)  
  Directory where the predictions file will be saved.  

- **`--base_dir`** (default: `PROJECT_ROOT`)  
  Base directory of the project. Useful for resolving relative paths.  

- **`--train_name`** (default: `""`)  
  Name of the experiment to load the trained model from (overrides `--model_path` if provided).  

- **`--out_var`** (default: `"SalePrice"`)  
  Name of the output variable/column in the predictions file.  

When the script is run, predictions a new directory will be created at `--output_dir` and the predictions will be saved to a CSV file in that directory along with a copy of the model and experiment configuration files.

## Running the API Locally (without Docker)

```bash
# Run the FastAPI app
uvicorn app.main:app --reload
```

Access the app at: http://localhost:8000

## Running the API with Docker

```bash
# Build the Docker image
docker build -t house-prices-app .

# Run the Docker container
docker run -rm -p 8000:8000 house-prices-app
```
Access the app at: http://localhost:8000

## API Documentation
The API documentation is automatically generated and can be accessed at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

API Endpoints
- `GET /` -> health check
- `GET /info/` -> get application info
- `POST /predict/` -> input data and get predictions