import numpy as np
import pandas as pd
from typing import Union, List, Tuple
import io
import os
import pathlib
from datetime import datetime, timedelta, time
import math
import time
import json
import random
from multiprocessing import Pool
from collections import deque




name_cryptocurrencies = ['_1', '_2', '_3']

def date_preprocessing(df: pd.DataFrame, labl: str = None) -> pd.DataFrame:
    df_c = df.copy()
    df_c['Full_time'] = df_c['Date'].astype('str') + df_c['Timestamp'].astype('str')
    df_c['Full_time'] = pd.to_datetime(df_c['Full_time'], format="%Y%m%d%H:%M:%S")
    df_c.drop(columns=['Timestamp', 'Date'], inplace=True)
    df_c = df_c.add_suffix(labl)
    df_c.rename(columns={f"Full_time{labl}": "Full_time"}, inplace=True)
    return df_c

def feature_engineering(df: pd.DataFrame, label: str = None) -> pd.DataFrame:
    df_c = df.copy()

    close = df_c['Close' + label]
    high = df_c['High' + label]
    low = df_c['Low' + label]
    open = df_c['Open' + label]

    # macd, _, macdhist = talib.MACD(close)
    # df_c['MACD' + label] = macd
    # df_c['EMA' + label] = talib.EMA(close, timeperiod=40)
    # slowk, slowd = talib.STOCH(high, low, close)
    # df_c['RSI' + label] = talib.RSI(close)
    df_c['DIFF' + label] = close.diff()
    df_c['BOP' + label] = (open - close) / (high - low)

    df_c['sin_weekday'] = np.sin(2 * np.pi * df_c['Full_time'].dt.weekday / 6)
    df_c['sin_hour'] = np.sin(2 * np.pi * df_c['Full_time'].dt.hour / 23)
    df_c['sin_minute'] = np.sin(2 * np.pi * df_c['Full_time'].dt.minute / 45)
    df_c['sin_days'] = np.sin(2 * np.pi * df_c['Full_time'].dt.day /
                                df_c['Full_time'].dt.days_in_month)
    return df_c


