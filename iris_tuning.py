"""
Nesse arquivo fazemos treinamento cruzado da base e verificação dos resultados obtidos
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


base = pd.read_csv('iris.csv')


previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4:5].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(optmizer, loss, activation, neurons, dropout,kernel_initializer):
    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation = activation,kernel_initializer=kernel_initializer, input_dim=4))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units=neurons, activation = activation))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units=3, activation = 'softmax'))
    classificador.compile(optimizer=optmizer, loss=loss,
                          metrics=['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede)
#optimizer: sgd
#activation: tanh
#kernel_initializer:random_normal
#
parametros={'batch_size':[10],
            'epochs':[1500],
            'optmizer':['adam','sgd'],
            'loss':['sparse_categorical_crossentropy'],
            'activation':['relu','tanh','selu'],
            'neurons':[8],
            'kernel_initializer':['random_uniform','random_normal','ones'],
            'dropout':[0.3]}
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           cv=4)
grid_search = grid_search.fit(previsores,classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_ 
