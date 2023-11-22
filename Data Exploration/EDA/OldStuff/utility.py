import pandas as pd

def readData(base, type = 1):
    if(type == 1):
        df_train_est = pd.read_csv(base + 'A/train_est_A.csv')
        df_train_est_A = df_train_est.rename(columns={'Unnamed: 0': 'date_forecast'})
        df_train_obs = pd.read_csv(base + 'A/train_obs_A.csv')
        df_train_obs_A = df_train_obs.rename(columns={'Unnamed: 0': 'date_forecast'})
        df_test = pd.read_csv(base + 'A/X_test_A.csv')
        df_test_A = df_test.rename(columns={'Unnamed: 0': 'date_forecast'})
        return df_train_est_A, df_train_obs_A, df_test_A
    elif(type == 2):
        df_train_est = pd.read_csv(base + 'B/train_est_B.csv')
        df_train_est_B = df_train_est.rename(columns={'Unnamed: 0': 'date_forecast'})
        df_train_obs = pd.read_csv(base + 'B/train_obs_B.csv')
        df_train_obs_B = df_train_obs.rename(columns={'Unnamed: 0': 'date_forecast'})
        df_test = pd.read_csv(base + 'B/X_test_B.csv')
        df_test_B = df_test.rename(columns={'Unnamed: 0': 'date_forecast'})
        return df_train_est_B, df_train_obs_B, df_test_B

    else:
        df_train_est = pd.read_csv(base + 'C/train_est_C.csv')
        df_train_est_C = df_train_est.rename(columns={'Unnamed: 0': 'date_forecast'})
        df_train_obs = pd.read_csv(base + 'C/train_obs_C.csv')
        df_train_obs_C = df_train_obs.rename(columns={'Unnamed: 0': 'date_forecast'})
        df_test = pd.read_csv(base + 'C/X_test_C.csv')
        df_test_C = df_test.rename(columns={'Unnamed: 0': 'date_forecast'})
        return df_train_est_C, df_train_obs_C, df_test_C
    
def remove_outliers(df):

    # Calcola il primo e terzo quartile per ciascuna colonna
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    # Calcola l'intervallo interquartile (IQR) per ciascuna colonna
    IQR = Q3 - Q1

    # Definisci i limiti per identificare gli outlier per ciascuna colonna
    limite_inferiore = Q1 - 1.5 * IQR
    limite_superiore = Q3 + 1.5 * IQR

    # Sostituisci gli outlier con i valori più estremi tra i non-outlier per ciascuna colonna
    for colonna in df.columns:
        df[colonna] = df[colonna].apply(lambda x: limite_inferiore[colonna] if x < limite_inferiore[colonna] else (limite_superiore[colonna] if x > limite_superiore[colonna] else x))
    return df

def get_df_train(df_est,df_obs):
    return pd.concat([df_obs, df_est], ignore_index=True)
    