import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from joblib import dump

# 读取训练数据
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# 前15列为输入数据
X_train = train_data.iloc[:, :-1].values
X_test = test_data.iloc[:, :-1].values

# 第16列为标签
y_train = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将标签转换为one-hot编码
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# 构建神经网络模型
model = Sequential()
model.add(Dense(64, input_dim=15, activation='relu', kernel_regularizer=l2(0.001)))  # 使用L2正则化
model.add(Dropout(0.5))  # 使用Dropout
model.add(Dense(4, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train_scaled, y_train_one_hot, epochs=1, batch_size=10, verbose=1, validation_data=(X_test_scaled, y_test_one_hot))

# 保存模型和标准化器
model.save('nn_model.keras')  # 使用推荐的Keras原生格式
dump(scaler, 'scaler.joblib')

# 评估模型
loss, accuracy = model.evaluate(X_test_scaled, y_test_one_hot)
print(f'Test accuracy: {accuracy}')
