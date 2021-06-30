import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# Separa as colunas que são formatadas em códigos (atomiza as colunas)

def oneHotEncoding(data, escolha):
    colunas = data.columns
    for i in range (len(colunas)):
        if(data[colunas[i]].dtypes == 'O'):
            enc = OneHotEncoder(handle_unknown = 'ignore')
            enc.fit(data[colunas[i]].values.reshape(-1, 1))
            if("Modelo 2" in escolha):
                sufixo = "_validado"
            else:
                sufixo = ""
            with open('pickles/modelos_tratamentos/'+"OHE_"+colunas[i]+sufixo+'.pk', 'wb') as fin:
                pickle.dump(enc, fin)
            dummies = pd.DataFrame(enc.transform(data[colunas[i]].values.reshape(-1, 1)).toarray())
            # adicionando o nome da coluna na dummie
            colunas_mais_nomes = [0]*dummies.shape[1]
            for k in range(dummies.shape[1]):
                colunas_mais_nomes[k] = colunas[i]+"-"+str(dummies.columns[k])
            dummies.columns = colunas_mais_nomes
            #concatenando a tabela de dummy criada com o dataset
            data = pd.concat([data,dummies],axis='columns')
            #dropando a antiga coluna
            data = data.drop(colunas[i],axis='columns')
    return data

def aplyOHE(data, escolha):
    colunas = data.columns
    for i in range (len(colunas)):
        if("Modelo 2" in escolha):
            sufixo = "_validado"
        else:
            sufixo = ""
        if(data[colunas[i]].dtypes == 'O'):
            with open('pickles/modelos_tratamentos/'+"OHE_"+colunas[i]+sufixo+'.pk', 'rb') as pickle_file:
                enc = pickle.load(pickle_file)
            dummies = pd.DataFrame(enc.transform(data[colunas[i]].values.reshape(-1, 1)).toarray())
            # adicionando o nome da coluna na dummie
            colunas_mais_nomes = [0]*dummies.shape[1]
            for k in range(dummies.shape[1]):
                colunas_mais_nomes[k] = colunas[i]+"-"+str(dummies.columns[k])
            dummies.columns = colunas_mais_nomes
            #concatenando a tabela de dummy criada com o dataset
            data = pd.concat([data,dummies],axis='columns')
            #dropando a antiga coluna
            data = data.drop(colunas[i],axis='columns')
    return data
    