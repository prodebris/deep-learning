from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model

# 构建模型
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))  # 确保指定输入形状
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 构建模型（可选，如果你已经通过传递数据调用了模型，则此步骤可以省略）
model.build(input_shape=(X_train.shape[1],))  # 使用输入数据的形状来构建模型

# 使用 plot_model 生成模型图
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)