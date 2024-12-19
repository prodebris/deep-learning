import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载训练数据和测试数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('testA.csv')

# 2. 处理缺失值（仅在训练数据中填充缺失值）
# 填充数值型特征的缺失值
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
# 使用 assign 方法添加列
test_data = test_data.assign(isDefault=0)

# 填充类别型特征的缺失值
categorical_cols = train_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])

# 3. 特征预处理
# 处理类别特征，进行 One-Hot 编码
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# 确保训练集和测试集的列一致
train_cols = train_data.columns
test_data = test_data.reindex(columns=train_cols, fill_value=0)

# 对数值特征进行标准化
numeric_features = ['loanAmnt', 'annualIncome', 'dti', 'installment', 'ficoRangeLow', 'ficoRangeHigh', 'delinquency_2years', 'openAcc', 'revolUtil', 'totalAcc']
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])

# 4. 特征选择
X_train = train_data.drop(['id', 'isDefault'], axis=1)  # 删除不必要的列（如ID和目标变量）
y_train = train_data['isDefault']  # 提取目标变量

# 确保X_train和X_test的列完全一致
X_test = test_data.drop(['id'], axis=1)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(train_data['isDefault'].value_counts())


# 5. 计算类别权重（处理类别不平衡）
# 将 [0, 1] 转换为 numpy 数组
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=train_data['isDefault'])
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# 6. 调整模型架构
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))  # 增加神经元
model.add(Dropout(0.3))  # 更高的Dropout比例
model.add(Dense(128, activation='relu'))  # 隐藏层1
model.add(Dropout(0.3))  # 更高的Dropout比例
model.add(Dense(64, activation='relu'))  # 隐藏层2
model.add(Dense(1, activation='sigmoid'))  # 输出层，sigmoid用于二分类问题

# 构建模型（可选，如果你已经通过传递数据调用了模型，则此步骤可以省略）
from tensorflow.keras.utils import plot_model
# 使用 plot_model 生成模型图
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# 7. 调整学习率
optimizer = Adam(learning_rate=0.001)  # 调整学习率

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])


# 假设 history 是 model.fit() 返回的训练历史
history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2, class_weight=class_weight_dict)

# 绘制训练和验证的损失
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('训练与验证损失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制AUC曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['AUC'], label='训练集AUC')
plt.plot(history.history['val_AUC'], label='验证集AUC')
plt.title('训练与验证AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

# 显示图像
plt.show()

# 10. 预测
y_pred = model.predict(X_test)

# 11. 生成提交文件
predictions = pd.DataFrame({
    'id': test_data['id'],
    'isDefault': y_pred.flatten()  # 预测结果为概率值
})

# 保存结果
predictions.to_csv('submission.csv', index=False)

print("预测结果已保存到 'submission.csv'")
