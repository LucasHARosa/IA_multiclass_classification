import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)


classificador = Sequential()
classificador.add(Dense(units=8, activation = 'tanh',kernel_initializer='random_normal', input_dim=4))
classificador.add(Dropout(0.3))
classificador.add(Dense(units=8, activation = 'tanh',kernel_initializer='random_normal'))
classificador.add(Dropout(0.3))
classificador.add(Dense(units=3, activation = 'softmax'))
classificador.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    
classificador.fit(previsores,classe, batch_size=10,epochs=1500)

classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')

novo = np.array([[6, 2.5, 5, 2.4]])
previsao = classificador.predict(novo)
previsao=(previsao>0.5)