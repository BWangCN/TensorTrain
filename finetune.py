import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# 加载已有的 Keras 模型
model = load_model('/Users/waynewang/Desktop/Sensor_Test/reskin_sensor/nn_prediction_model5.keras')

# 选择性冻结部分层（如果不需要冻结，可以跳过这一步）
for layer in model.layers[:-1]:  # 冻结除最后一层外的所有层
    layer.trainable = False

# 加载新的训练数据
new_data = pd.read_csv('/Users/waynewang/Desktop/Sensor_Test/reskin_sensor/train_data9.csv')

# 假设前15列是特征，最后一列是标签
X_new = new_data.iloc[:, :-1].values
y_new = new_data.iloc[:, -1].values

# 如果之前的数据进行了标准化处理，这里也需要同样的处理
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# 将标签转换为 one-hot 编码
y_new_one_hot = to_categorical(y_new, num_classes=4)  # 确保 num_classes 和输出层数量一致

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
model.fit(X_new_scaled, y_new_one_hot, epochs=10, batch_size=8, verbose=1)

# 保存更新后的模型
model.save('updated_model_5.keras')

# 如果需要保存标准化器，也可以使用 joblib 进行保存
from joblib import dump
dump(scaler, 'scaler.joblib')

