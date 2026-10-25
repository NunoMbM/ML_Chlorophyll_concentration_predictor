from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATE = "date"
TARGET = "chlorophyll_a_mg_m3"
'''
DEFAULT_LAGS = {
    TARGET: list(range(1, 15)),
    "sst_c": list(range(0, 15)),
    "par_umol_m2_s": list(range(0, 8)),
    "nitrate_umol_l": list(range(0, 15)),
    "wind_speed_m_s": list(range(0, 8)),
    "upwelling_index": list(range(0, 15)),
    "mixed_layer_depth_m": list(range(0, 8)),
    "salinity_psu": list(range(0, 8)),
    "current_speed_m_s": list(range(0, 8)),
    "river_discharge_index": list(range(0, 8)),
    "cloud_fraction": list(range(0, 8)),
    "surface_pressure_hpa": list(range(0, 8)),
    "turbidity_ntu": list(range(0, 8)),
}
'''
DEFAULT_LAGS = {
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



def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="daily_data") if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df[DATE] = pd.to_datetime(df[DATE])
    return df.sort_values(DATE).reset_index(drop=True)


def build_lagged_frame(raw: pd.DataFrame, lag_spec: dict[str, list[int]] | None = None,
                       require_target: bool = True) -> pd.DataFrame:
    """Return rows indexed by prediction date, never using target lag 0."""
    lag_spec = lag_spec or DEFAULT_LAGS
    if DATE not in raw:
        raise ValueError(f"missing required column: {DATE}")
    if require_target and TARGET not in raw:
        raise ValueError(f"missing required column: {TARGET}")
    frame = raw.copy().sort_values(DATE).reset_index(drop=True)
    columns: dict[str, pd.Series | np.ndarray] = {DATE: pd.to_datetime(frame[DATE])}
    if TARGET in frame:
        columns[TARGET] = frame[TARGET].astype(float)
    for col, lags in lag_spec.items():
        if col not in frame:
            raise ValueError(f"lag specification refers to missing column: {col}")
        for lag in lags:
            if col == TARGET and lag == 0:
                raise ValueError("target lag 0 is forbidden: it leaks the answer")
            columns[f"{col}__lag_{lag}"] = pd.to_numeric(frame[col], errors="coerce").shift(lag)
    day = pd.to_datetime(frame[DATE]).dt.dayofyear.to_numpy()
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)
    out = pd.DataFrame(columns)
    if require_target:
        out = out.dropna(subset=[TARGET])
    max_lag = max(max(v) for v in lag_spec.values())
    return out.iloc[max_lag:].reset_index(drop=True)


def one_day_row(history: pd.DataFrame, day_features: pd.DataFrame,
                lag_spec: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Construct exactly one feature row from observed history plus today's exogenous data."""
    if len(day_features) != 1:
        raise ValueError("day_features must contain exactly one row")
    today = day_features.copy()
    if TARGET not in today:
        today[TARGET] = np.nan
    combined = pd.concat([history, today], ignore_index=True, sort=False)
    lagged = build_lagged_frame(combined, lag_spec=lag_spec, require_target=False)
    row = lagged.tail(1).drop(columns=[DATE, TARGET], errors="ignore")
    if len(row) != 1:
        raise ValueError("insufficient history to construct the requested lags")
    return row


def split_data(df, target_column, train_size=0.8, val_size=0.1):

  df_sorted = df.sort_values(by='date', ascending=True)

  size = len(df_sorted)

  size = len(df_sorted)
  size_train = int(size * train_size)
  size_val = int(size * val_size)

  df_train = df_sorted.iloc[:int(size_train)]
  df_validation = df_sorted.iloc[int(size_train):int(size_val+size_train)]
  df_test = df_sorted.iloc[int(size_val+size_train):]

  cols_to_drop = [target_column]
  if 'date' in df.columns:
      cols_to_drop.append('date')

  X_train = df_train.drop(columns=cols_to_drop)
  Y_train = df_train[target_column]
  X_test = df_test.drop(columns=cols_to_drop)
  Y_test = df_test[target_column]

  if size_val != 0:
    X_valitation = df_validation.drop(columns=[cols_to_drop])
    Y_valitation = df_validation[cols_to_drop]
    return X_train, Y_train, X_valitation, Y_valitation, X_test, Y_test
  else:
    return X_train, Y_train, X_test, Y_test

#metricas de performance
def assessment(X, y, model,imprimir=True):
  y_pred = model.predict(X)

  r2 = r2_score(y, y_pred)
  mae = mean_absolute_error(y, y_pred)
  mse = mean_squared_error(y, y_pred)
  rmse = np.sqrt(mse)

  if imprimir:
        print(f"R² Score: {r2:.4f}")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"Raiz do Erro Quadrático (RMSE): {rmse:.4f}")

  return r2, mae, mse, rmse


#testar diferentes janelas
def folds_recentes(df, target_column, n_splits=5, model=None, pesos=None, scaler = False):
    if model is None:
        print("Nenhum modelo selecionado. default: LinearRegression")
        model = LinearRegression()

    df_sorted = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    tamanho_bloco = len(df_sorted) // n_splits


    inicio_teste = len(df_sorted) - tamanho_bloco
    test_df = df_sorted.iloc[inicio_teste:]

    cols_to_drop = [target_column]
    if 'date' in df.columns:
        cols_to_drop.append('date')

    X_test = test_df.drop(columns=cols_to_drop)
    Y_test = test_df[target_column]

    r2_vec, mae_vec, mse_vec, rmse_vec = [], [], [], []

    for i in range(1, n_splits):
        inicio_treino = inicio_teste - (i * tamanho_bloco)


        if inicio_treino < 0:
            inicio_treino = 0

        train_df = df_sorted.iloc[inicio_treino:inicio_teste]

        X_train = train_df.drop(columns=cols_to_drop)
        Y_train = train_df[target_column]

        if scaler:

            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)

            X_test = std_scaler.transform(X_test)


        if pesos is not None:
            n_amostras = len(X_train)
            sample_weights = np.linspace(start=pesos[0], stop=pesos[1], num=n_amostras)
            model.fit(X_train, Y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, Y_train)

        data_inicio = df_sorted['date'].iloc[inicio_treino].date()
        data_fim = df_sorted['date'].iloc[inicio_teste - 1].date()
        #print(f"Fold {i}: Treino usa dados de {data_inicio} a {data_fim}")

        r2, mae, mse, rmse = assessment(X_test, Y_test, model, imprimir=False)
        r2_vec.append(r2)
        mae_vec.append(mae)
        mse_vec.append(mse)
        rmse_vec.append(rmse)

        if inicio_treino == 0:
            break

    #print(f"\nMax R²: {np.max(r2_vec):.4f}")
    #print(f"Min RMSE: {np.min(rmse_vec):.4f}")

    return r2_vec, mae_vec, mse_vec, rmse_vec


def create_engineered_features(df):
    """Cria novas variáveis com base no conhecimento do domínio, tendências e volatilidade."""
    df_eng = df.copy()
    
    # 1. Interações (Efeito multiplicativo)
    if 'par_umol_m2_s' in df_eng.columns and 'nitrate_umol_l' in df_eng.columns:
        df_eng['luz_x_nutrientes'] = df_eng['par_umol_m2_s'] * df_eng['nitrate_umol_l']
        
    if 'par_umol_m2_s' in df_eng.columns and 'upwelling_index' in df_eng.columns:
        df_eng['luz_x_upwelling'] = df_eng['par_umol_m2_s'] * df_eng['upwelling_index']
        
    # 2. Estatísticas Móveis (Tendência e Acumulação)
    # A janela de 7 e 3 dias ajuda a captar a dinâmica da semana anterior
    for col in ['sst_c', 'wind_speed_m_s', 'par_umol_m2_s']:
        if col in df_eng.columns:
            df_eng[f'{col}_roll_mean_7'] = df_eng[col].rolling(window=7, min_periods=1).mean()
            df_eng[f'{col}_roll_std_3'] = df_eng[col].rolling(window=3, min_periods=1).std().fillna(0)
            
    # 3. Deltas (Taxa de variação diária - Aquecimento/Arrefecimento)
    for col in ['sst_c', 'surface_pressure_hpa', 'mixed_layer_depth_m']:
        if col in df_eng.columns:
            df_eng[f'{col}_diff_1'] = df_eng[col].diff().fillna(0)
            
    return df_eng

