# -*- coding: utf-8 -*-
import sys
import pickle
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from conexaoDados import range_dados
from datetime import date, timedelta
from conexaoDados import todos_dados
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split
from tratarDados import refinamento_hiperparametros

def avaliacao(data_treino, data_teste, label_treino,label_teste, hiperparametros):
    modelo = SVC(kernel="linear", random_state=0)
    for i in range(len(hiperparametros["C"])):
        modelo = SVC(kernel="linear", random_state=0, C = hiperparametros["C"][i])
        # Treina o modelo de predicao da classe
        modelo.fit(data_treino, label_treino.values.ravel())
        y_predito = modelo.predict(data_teste)
        micro = f1_score(label_teste,y_predito,average='micro')
        macro = f1_score(label_teste,y_predito,average='macro')
        string = ""
        valor_c = modelo.C
        print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"é: ",micro)
        print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"é: ",macro)
    print("")

data = pd.read_csv("../ProjetoTCE/arquivos/dadosTCE.csv",low_memory = False)
data.drop("Empenho (Sequencial Empenho)(EOF).1",axis = 1,inplace= True)
data.columns = [data.columns[i].replace("(EOF)","") for i in range(len(data.columns))]
dados_validados = pd.read_excel("dados_analisados.xlsx")
# Retirando os dados analisados do conjunto principal
indexes = [0]*dados_validados.shape[0]
for i in range(dados_validados.shape[0]):
    indexes[i] = data['Empenho (Sequencial Empenho)'][data['Empenho (Sequencial Empenho)'] == dados_validados['Empenho (Sequencial Empenho)(EOF)'].iloc[i]].index[0]
data.drop(indexes,inplace = True)
data.reset_index(drop = True, inplace = True)
del indexes
#
data = data[:10000]
dados_validados = dados_validados[:100]
print(data.shape)
tratamentoDados(data.copy(),'tfidf')
tratamentoDados(data.copy(),'OHE')
data = pickles.carregarPickle("data")
label = pickles.carregarPickle("label")
tfidf = pickles.carregarPickle("tfidf")
data = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
del tfidf
print(data.shape)
# Separando 40% dos dados para selecao de hiperparametros
data_treino, data_teste, label_treino, label_teste = train_test_split(data, label, test_size = 0.6,stratify = label, random_state = 10)
data_treino, data_teste, label_treino, label_teste = train_test_split(data_treino, label_treino, test_size = 0.3,stratify = label_treino, random_state = 10)
label_treino.reset_index(drop = True, inplace = True)
label_teste.reset_index(drop = True, inplace = True)
# Acha o melhor conjunto de hiperparametros para o algoritmo
hiperparametros = {'C':[0.1,1,10,50] }
#hiperparametros = {'C':[] }
avaliacao(data_treino, data_teste, label_treino,label_teste,hiperparametros)


# Tratando os dados validados
label_validado = dados_validados["ANÁLISE"]
dados_validados.drop("ANÁLISE",inplace = True,axis = 1)
# Tratando os dados validados
data_validado, label_naturezas = tratarDados(dados_validados.copy(),'OHE')
tfidf_validado, label_naturezas = tratarDados(dados_validados.copy(),'tfidf')
data_validado = sparse.hstack((csr_matrix(data_validado),csr_matrix(tfidf_validado) ))
del tfidf_validado, dados_validados, label_naturezas
print(data_validado.shape)
# Treina o modelo de predicao de corretude    
# Separando 40% dos dados para treino e 40% para teste
data_treino, data_teste, label_treino, label_teste = train_test_split(data_validado, label_validado, test_size = 0.6,stratify = label_validado, random_state = 10)
label_treino.reset_index(drop = True, inplace = True)
label_teste.reset_index(drop = True, inplace = True)
# Acha o melhor conjunto de hiperparametros para o algoritmo
hiperparametros = {'C':[0.1,1,10,50] }
#hiperparametros = {'C':[] }
avaliacao(data_treino, data_teste, label_treino,label_teste,hiperparametros)
