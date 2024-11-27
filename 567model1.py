import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, r2_score

data = pd.read_csv('C:\\Users\\aedyn\\Downloads\\houses8.csv')

x = data.drop('PRICE', axis=1)
y = data['PRICE']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=80)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(8, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
model.fit(x_train, y_train, batch_size = 1, epochs = 200)

pred = model.predict(x_test)
print('Variance Score: ', explained_variance_score(y_test, pred))
print('R2 Score: ', r2_score(y_test, pred))
model.evaluate(x_test, y_test, verbose = 2)

ex_house = scaler.transform(np.array([3, 2, 1653, 3929, 2000, 3, 35.159596353067734, -106.60611419469532]).reshape(-1, 8))
print(model.predict(ex_house)[0,0])
