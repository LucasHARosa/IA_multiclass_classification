"""
Nesse arquivo fazemos uma divisão da base e verificação dos resultados obtidos
"""
# importação para estrutura da rede neural
from keras.models import Sequential
from keras.layers import Dense

# leitura da base
import pandas as pd
base = pd.read_csv('iris.csv')

# Divisão de previsores e classe
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4:5].values

# Para transformar categoria em número
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

# Para colocar no formato das classes certo, 
#exemplo:   iris setosa=100
#           iris virginica = 010  
#           iris versicolor = 001 
from keras.utils import np_utils
classe_dummy = np_utils.to_categorical(classe)

# importação para dividir a base em treino e teste
from sklearn.model_selection import train_test_split

# Divisão entre treino e teste
previsores_treinamento, previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)
# Criação de uma rede neural do tipo sequencial
classificador = Sequential()
# 1° camada oculta
classificador.add(Dense(units=4, activation = 'relu', input_dim=4))
# 2° camada oculta
classificador.add(Dense(units=4, activation = 'relu'))
# camada de saída
"""
Parametros:
    activation: Função de ativação
        softmax: Softmax é freqüentemente usado como ativação para a última camada
        de uma rede de classificação porque o resultado pode ser 
        interpretado como uma distribuição de probabilidade.
        exp(x)/tf.reduce_sum(exp(x))
"""
classificador.add(Dense(units=3, activation = 'softmax'))

"""
Parametros:
    optimizer: padrão de atualização
        adam: É um método de descida de gradiente estocástico baseado na estimativa adaptativa de momentos de primeira e segunda ordem
    loss: função de perda ou erro
        categorical_crossentropy: Use esta função de perda de entropia cruzada quando houver duas ou mais classes de rótulos 
    metric: Metrica de avaliação
        categorical_accuracy:
"""
classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
# treinamento da rede
classificador.fit(previsores_treinamento,classe_treinamento, batch_size=10,
                  epochs=1000)

# Visualização de resultados
resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5)

# comparação na matriz sobre os resultados
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)