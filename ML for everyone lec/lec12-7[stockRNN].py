### stock daily RNN-LSTM ###

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn

tf.set_random_seed(777)
tf.reset_default_graph()

## parameter ##
seq_length = 15
data_dim = 5
output_dim = 1
hidden_size = 10
learning_rate = 0.01
iteration = 2000

### 데이터 전처리 ###
stock_data = np.loadtxt("d:/data/prac/stock_daily.csv", delimiter=',', skiprows=1)
stock_data = stock_data[::-1] # shape (732, 5)

## split ## --> 시계열(시간순)
train_size = int(len(stock_data) * 0.7)
train_set = stock_data[:train_size, :] # shape(512, 5)
test_set = stock_data[train_size:, :] # test(220, 5)

# RNN data building #
def build(time_series, seq_length):
    x_data = []
    y_data = []
    for i in range(0, len(time_series) - seq_length):
        x_tmp = time_series[i: i + seq_length, :]
        y_tmp = time_series[i + seq_length, [-1]]
        x_data.append(x_tmp)
        y_data.append(y_tmp)
    return np.array(x_data), np.array(y_data)





















