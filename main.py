import pandas as pd
import numpy as np
import itertools
from data_preproc import name_cryptocurrencies, date_preprocessing, feature_engineering
from agent import Play_Agent
from trading_environment import TradingEnv


# Загрузка данных стоимости BTC-USD
df_1 = pd.read_csv('BTCUSD.csv')
# Загрузка данных стоимости ETH-USD
df_2 = pd.read_csv('ETHUSD.csv')
# Загрузка данных стоимости LTC-USD
df_3 = pd.read_csv('LTCUSD.csv')

for df, num in zip([df_1, df_2, df_3], name_cryptocurrencies):
    if num == '_1':
        df_ALL = date_preprocessing(df, num)
    else:
        df_ALL = df_ALL.merge(date_preprocessing(df, num), on='Full_time')

for lb in name_cryptocurrencies:
    df_ALL = feature_engineering(df_ALL, label=lb)

# Добавление корреляция
df_ALL['Corr_1_2'] = df_ALL[['Close_1', 'Close_2']].rolling(96).corr().loc[::2, 'Close_2'].reset_index(drop=True)
df_ALL['Corr_1_3'] = df_ALL[['Close_1', 'Close_3']].rolling(96).corr().loc[::2, 'Close_3'].reset_index(drop=True)
df_ALL['Corr_2_3'] = df_ALL[['Close_2', 'Close_3']].rolling(96).corr().loc[::2, 'Close_3'].reset_index(drop=True)

drop_columns = list(''.join(i) for i in itertools.product(['High', 'Low', 'Open'], name_cryptocurrencies))
df_ALL.dropna(inplace=True)
df_ALL.drop(columns=drop_columns, inplace=True)
df_ALL.reset_index(drop=True, inplace=True)

# Интервал времени под который есть преобразованные данные ceemdan
df_ALL_train = df_ALL[(df_ALL['Full_time'] > '2020-02-06') & (df_ALL['Full_time'] < '2021-02-06')]

# Загрузка предобразованных данных с помощью EMD-signal
ceemdan_1_train = np.load('Close_1_ceemdan.npy')
ceemdan_2_train = np.load('Close_2_ceemdan.npy')
ceemdan_3_train = np.load('Close_3_ceemdan.npy')
ceemdan_all_train = np.vstack((
    np.expand_dims(ceemdan_1_train, axis=0),
    np.expand_dims(ceemdan_2_train, axis=0),
    np.expand_dims(ceemdan_3_train, axis=0)
    ))

# Создание торгового агента
work_agent = Play_Agent(
    TradingEnv(df_ALL_train, ceemdan_all_train)
)
# Запуск обучения
work_agent.play_episod(100)