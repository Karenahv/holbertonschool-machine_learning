#!/usr/bin/env python3
"""Preprocess data"""

import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)


def preprocessor(data):
    """
    :param data: Dataset
    :return: data preprocessed
    """
    data = pd.read_csv(data)
    data['Volume_(BTC)'].fillna(value=0, inplace=True)
    data['date'] = pd.to_datetime(data['Timestamp'], unit='s')
    data['Volume_(Currency)'].fillna(value=0, inplace=True)
    data['Weighted_Price'].fillna(value=0, inplace=True)

    # next we need to fix the OHLC
    # (open high low close) data which
    # is a continuous timeseries so
    # lets fill forwards those values...
    data['Open'].fillna(method='ffill', inplace=True)
    data['High'].fillna(method='ffill', inplace=True)
    data['Low'].fillna(method='ffill', inplace=True)
    data['Close'].fillna(method='ffill', inplace=True)

    return data
