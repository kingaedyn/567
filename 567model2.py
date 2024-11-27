import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from sklearn.metrics import explained_variance_score

df = pd.read_csv('C:\\Users\\aedyn\\Downloads\\houses8.csv')

x = df.drop('PRICE', axis=1)
y = df['PRICE']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 1, 8)
x_test  = x_test.reshape(-1, 1, 8)

print(x_train.shape)
print(x_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(1, 8), return_sequences=True))
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
model.fit(x_train, y_train, batch_size = 8, epochs = 400)

pred = model.predict(x_test)
print('Variance Score: ', explained_variance_score(y_test, pred))
model.evaluate(x_test, y_test, verbose = 2)

# ex_house = scaler.transform(np.array([3, 2, 1653, 3929, 2000, 3, 35.159596353067734, -106.60611419469532]).reshape(-1, 8))
# print(model.predict(ex_house)[0,0])
