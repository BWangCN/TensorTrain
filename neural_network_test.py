import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.metrics import accuracy_score

# 加载测试数据
test_data = pd.read_csv('test_data.csv')

# 前15列为输入数据
X_test = test_data.iloc[:, :-1].values

# 第16列为标签
y_test = test_data.iloc[:, -1].values

# 加载标准化器
scaler = load('scaler.joblib')

# 使用训练好的标准化器对测试数据进行标准化
X_test_scaled = scaler.transform(X_test)

# 加载训练好的模型
model = load_model('nn_prediction_model4.keras')

# 使用模型进行预测
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
