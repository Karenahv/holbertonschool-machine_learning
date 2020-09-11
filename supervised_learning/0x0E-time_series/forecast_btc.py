#!/usr/bin/env python3
""" Time Series Forecasting for bitcoin"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras as K


def forecast(data):
    """
    :param data: Dataset
    :return:
    """
    group = data.groupby('date')
    Real_Price = group['Weighted_Price'].mean()

    # split data
    prediction_days = 30
    df_train = Real_Price[len(Real_Price) - prediction_days:]
    df_test = Real_Price[:len(Real_Price) - prediction_days]

    # Data preprocess
    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set) - 1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))

    # Initialising the RNN
    regressor = K.models.Sequential()

    # Adding the input layer and the LSTM layer
    regressor.add(K.layers.LSTM(units=128, activation='sigmoid',
                                input_shape=(None, 1)))

    # Adding the output layer
    regressor.add(K.layers.Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, batch_size=5, epochs=1000)

    regressor.summary()

    # Making the predictions
    test_set = df_test.values
    inputs = np.reshape(test_set, (len(test_set), 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted_BTC_price = regressor.predict(inputs)
    predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

    # Visualising the results
    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.plot(test_set, color='red', label='Real BTC Price')
    plt.plot(predicted_BTC_price, color='blue', label='Predicted BTC Price')
    plt.title('BTC Price Prediction', fontsize=14)
    df_test = df_test.reset_index()
    x = df_test.index
    labels = df_test['date']
    plt.xticks(x, labels, rotation='vertical')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('BTC Price(USD)', fontsize=14)
    plt.legend(loc=2, prop={'size': 14})
    plt.show()


if __name__ == '__main__':
    preprocess = __import__('preprocess_data').preprocessor

    data = preprocess('../input/coinbaseUSD_1-min_'
                      'data_2014-12-01_to_2019-01-09.csv')

    forecast(data)
