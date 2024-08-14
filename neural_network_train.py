import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from joblib import dump

# 读取训练数据
train_data1 = pd.read_csv('/Users/waynewang/Desktop/Sensor_Test/reskin_sensor/dataset2/train_data1.csv')
train_data2 = pd.read_csv('/Users/waynewang/Desktop/Sensor_Test/reskin_sensor/dataset2/train_data2.csv')
train_data3 = pd.read_csv('/Users/waynewang/Desktop/Sensor_Test/reskin_sensor/dataset2/train_data3.csv')

# 合并两个数据集
train_data = pd.concat([train_data1, train_data2, train_data3], axis=0)

# 前15列为输入数据
X_train = train_data.iloc[:, :-1].values

# 第16列为标签
y_train = train_data.iloc[:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 将标签转换为one-hot编码
y_train_one_hot = to_categorical(y_train)

# 构建神经网络模型
model = Sequential()
model.add(Dense(64, input_dim=15, activation='tanh'))  # 第一层
model.add(Dense(32, activation='tanh'))                # 第二层
model.add(Dense(4, activation='softmax'))              # 输出层

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train_scaled, y_train_one_hot, epochs=1, batch_size=10, verbose=1)

# 保存模型和标准化器
model.save('nn_model/nn_model1.keras')
dump(scaler, 'scaler.joblib')


