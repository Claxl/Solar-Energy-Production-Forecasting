import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_df(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled

def readData(location):
  if(location == 1):
    y_train_obs_A = pd.read_parquet('data/A/train_targets.parquet')
    X_test_est_A = pd.read_parquet('data/A/X_test_estimated.parquet')
    X_train_est_A = pd.read_parquet('data/A/X_train_estimated.parquet')
    X_train_obs_A = pd.read_parquet('data/A/X_train_observed.parquet')
    X_a = pd.concat([X_train_est_A, X_train_obs_A])
    return y_train_obs_A, X_test_est_A, X_a
  elif(location == 2):
    y_train_obs_B = pd.read_parquet('data/B/train_targets.parquet')
    X_test_est_B = pd.read_parquet('data/B/X_test_estimated.parquet' )
    X_train_est_B = pd.read_parquet('data/B/X_train_estimated.parquet')
    X_train_obs_B = pd.read_parquet('data/B/X_train_observed.parquet')
    X_b = pd.concat([X_train_est_B, X_train_obs_B])
    return y_train_obs_B, X_test_est_B, X_b
  else:
    y_train_obs_C = pd.read_parquet('data/C/train_targets.parquet')
    X_test_est_C = pd.read_parquet('data/C/X_test_estimated.parquet')
    X_train_est_C = pd.read_parquet('data/C/X_train_estimated.parquet')
    X_train_obs_C = pd.read_parquet('data/C/X_train_observed.parquet')
    X_c = pd.concat([X_train_est_C, X_train_obs_C])
    return y_train_obs_C, X_test_est_C, X_c

def resize_training_data(X_train, y_train):
    y_features = y_train.columns.tolist()
    X_date_feature = "date_forecast"
    X_date_calc_feature = "date_calc"
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature, X_date_calc_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized


