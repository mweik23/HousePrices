# House Prices Regressor
An application that predicts house prices based on a set of input features.

## Overview
This project is a FastAPI-based web application that provides an API for predicting house prices. It is containerized with Docker for easy deployment and reproducibility. Disclaimer: this is an educational project. This tool is not going to be very effective at predicting house prices that are on the market now because it is based on a machine learning model that was trained using data from several cities around the years of 2007-2008.

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
pip install -r requirements-app.txt

# Run the FastAPI app
uvicorn app.main:app --reload
```

Access the app at: http://localhost:8000

## Running with Docker

```bash
# Build the Docker image
docker build -t house-prices-app .

# Run the Docker container
docker run -d -p 8000:8000 house-prices-app
```
Access the app at: http://localhost:8000