from catboost import CatBoostRegressor

# Assume you’ve loaded:
#   X_train, y_train    → shapes (n_train × 226), (n_train,)
#   X_val,   y_val      → shapes (n_val   × 226), (n_val,)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    bagging_temperature=1.0,
    random_seed=42,
    task_type="CPU",         # or "GPU" if you have one
    od_type="Iter",          # “Iter” means use early stopping
    od_wait=50               # stop after 50 rounds without improvement on val
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    verbose=100
)

# After fitting, you can check metrics on validation:
val_preds = model.predict(X_val)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")
