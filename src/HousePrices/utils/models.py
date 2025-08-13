from dataclasses import dataclass
from typing import Literal, Dict, Any
from catboost import CatBoostRegressor

ModelName = Literal["catboostregressor"]

@dataclass
class ModelConfig:
    name: ModelName
    params: Dict[str, Any]  # hyperparams for the chosen model

def build_model(cfg: ModelConfig):
    if cfg.name == "catboostregressor":
        return CatBoostRegressor(**cfg.params)
    raise ValueError(f"Unknown model: {cfg.name}")

'''
For now, only CatBoostRegressor is implemented. More models can be added as needed. A good starting point for CatBoostRegressor is
given below:

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    bagging_temperature=1.0,
    random_seed=42,
    task_type="CPU",         # or "GPU" if you have one
    od_type="Iter",          # “Iter” means use early stopping
    train_dir=expdir,
    od_wait=50               # stop after 50 rounds without improvement on val
)
'''

