import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import read_table, build_lagged_frame,split_data, create_engineered_features, assessment,folds_recentes
import joblib

data = read_table("chlorophyll_student_2015_2023.csv")
#g = sns.PairGrid(data)
#g.map(sns.scatterplot)

data_eng = create_engineered_features(data)
#g = sns.PairGrid(data_eng)
#g.map(sns.scatterplot)

TARGET = "chlorophyll_a_mg_m3"
lag_alterados = {
    TARGET: [1, 2, 3],
    "sst_c": [0, 1, 2, 3],
    "par_umol_m2_s": [0, 1, 2, 3],
    "nitrate_umol_l": [0, 1, 2, 3],
    "wind_speed_m_s": [0, 1, 2, 3],
    "upwelling_index": [0, 1, 2, 3],
    "mixed_layer_depth_m": [0, 1, 2, 3],

    "luz_x_nutrientes": [0, 1, 2, 3],
    "sst_c_roll_mean_7": [0],
    "sst_c_diff_1": [0, 1, 2],
}


df_default = build_lagged_frame(data)
df_alterados = build_lagged_frame(data_eng, lag_spec=lag_alterados)

lr_model = LinearRegression()
ridge_model = RidgeCV(alphas=(0.1, 1.0, 10.0))

base_rf = RandomForestRegressor(random_state=42)




print(" Defaults")
print("===============================================")
r2_lr_def, _, _, rmse_lr_def = folds_recentes(df_default, TARGET, n_splits=4, model=lr_model, scaler=True)
print(f"-> Regressão Linear | Max R²: {max(r2_lr_def):.4f} | Min RMSE: {min(rmse_lr_def):.4f}")

r2_rid_def, _, _, rmse_rid_def = folds_recentes(df_default, TARGET, n_splits=4, model=ridge_model, scaler=True)
print(f"-> Ridge     | Max R²: {max(r2_rid_def):.4f} | Min RMSE: {min(rmse_rid_def):.4f}")

r2_log_def, _, _, rmse_log_def = folds_recentes(df_default, TARGET, n_splits=4, model=base_rf, scaler=False)
print(f"-> RandomForestRegressor    | Max R²: {max(r2_log_def):.4f} | Min RMSE: {min(rmse_log_def):.4f}")


print("\n")
print("Lags alterados + features")
print("===============================================")
r2_lr_dyn, _, _, rmse_lr_dyn = folds_recentes(df_alterados, TARGET, n_splits=4, model=lr_model, scaler=True)
print(f"-> Regressão Linear| Max R²: {max(r2_lr_dyn):.4f} | Min RMSE: {min(rmse_lr_dyn):.4f}")

r2_rig_dyn, _, _, rmse_rig_dyn = folds_recentes(df_alterados, TARGET, n_splits=4, model=ridge_model, scaler=True)
print(f"-> Ridge     | Max R²: {max(r2_rig_dyn):.4f} | Min RMSE: {min(rmse_rig_dyn):.4f}")

r2_log_dyn, _, _, rmse_log_dyn = folds_recentes(df_alterados, TARGET, n_splits=4, model=base_rf, scaler=False)
print(f"-> RandomForestRegressor     | Max R²: {max(r2_log_dyn):.4f} | Min RMSE: {min(rmse_log_dyn):.4f}")

print("\n"+"="*50+"\n")


X_train, y_train, X_test, y_test = split_data(df_alterados, target_column=TARGET, train_size=0.80, val_size=0)

param_grid_ = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

tscv = TimeSeriesSplit(n_splits=4)

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid_,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)

print("hiperparâmetros")
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print(f"parâmetros :\n{grid_search.best_params_}\n")

assessment(X_test, y_test, model, imprimir=True)


joblib.dump(model, "model.pkl")
print("Modelo guardado")
