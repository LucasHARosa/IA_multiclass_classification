"""
Nesse arquivo fazemos treinamento cruzado da base e verificação dos resultados obtidos
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


base = pd.read_csv('iris.csv')


previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4:5].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criar_rede():
    classificador = Sequential()
    """
    Parametros:
        activation: Função de ativação
            tanh: Função de ativação da tangente hiperbólica.
            softmax: Softmax converte um vetor de valores em uma distribuição de probabilidade. Os elementos do vetor de saída estão no intervalo (0, 1) e somam 1.
        kernel_initializer: Inicialização dos pesos
            random_normal: 
    """
    classificador.add(Dense(units=8, activation = 'tanh',kernel_initializer='random_normal', input_dim=4))
    classificador.add(Dropout(0.3))
    classificador.add(Dense(units=8, activation = 'tanh',kernel_initializer='random_normal'))
    classificador.add(Dropout(0.3))
    classificador.add(Dense(units=3, activation = 'softmax'))
    """
    Parametros compile:
        optimizer: padrão de atualização
            sgd: Otimizador de descida de gradiente (com impulso)
        loss: função de perda ou erro
            sparse_categorical_crossentropy: Calcula com que frequência as previsões correspondem a rótulos inteiros.
        metrics: Metrica de avaliação
            accuracy: Calcula com que frequência as previsões são iguais aos rótulos.
    """
    classificador.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    return classificador
    
classificador = KerasClassifier(build_fn=criar_rede,epochs=1500, batch_size=10)

resultados = cross_val_score(estimator=classificador, X=previsores,y=classe,
                            cv=10, scoring='accuracy')
media = resultados.mean()
desvio=resultados.std()
