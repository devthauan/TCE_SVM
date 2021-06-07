### Bibliotecas python ###
import pickle
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
### Meus pacotes ###
from tratamentos import pickles
from tratamentos import tratar_texto
from tratamentos import one_hot_encoding
from preparacaoDados import tratamento_especifico
### Meus pacotes ###

def tratarDados(data, opcao):
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    identificador_empenho = pd.DataFrame(data['empenho_sequencial_empenho'])
    pickles.criarPickle(identificador_empenho,"modelos_tratamentos/identificador_empenho")
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['exercicio_do_orcamento_ano','classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome',
                      'valor_estorno_anulacao_empenho','valor_anulacao_cancelamento_empenho',
                      'fonte_recurso_cod','elemento_despesa','grupo_despesa',
                      'empenho_sequencial_empenho'], axis='columns')
    # rotulo
    label = data['natureza_despesa_cod']
    label = pd.DataFrame(label)
    data = data.drop('natureza_despesa_cod',axis = 1)
    # tfidf
    textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
    # Função que gera o TF-IDF do texto tratado
    with open('pickles/modelos_tratamentos/tfidf_modelo'+'.pk', 'rb') as pickle_file:
        tfidf_modelo = pickle.load(pickle_file)
    tfidf =  pd.DataFrame.sparse.from_spmatrix(tfidf_modelo.transform(textoTratado))
    del textoTratado
    data = data.drop('empenho_historico',axis = 1)
    # Tratamento dos dados
    data = tratamento_especifico(data.copy())
    # Normalizando colunas numéricas
    colunas = data.columns
    for col in colunas:
        if(data[col].dtype != "O"):
            with open('pickles/modelos_tratamentos/'+"normalization_"+col+'.pk', 'rb') as pickle_file:
                min_max_scaler = pickle.load(pickle_file)
            data[col] = pd.DataFrame(min_max_scaler.transform(data[col].values.reshape(-1,1)))
    # OHE
    data = one_hot_encoding.aplyOHE(data)
    if(opcao == "OHE"):
        return data, label
    elif(opcao == "tfidf"):
        return tfidf, label
    else:
        aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
        data =  pd.DataFrame.sparse.from_spmatrix(aux)
        return data, label


def refinamento_hiperparametros(data_treino, label_treino, modelo , hiperparametros, espalhamento):
    # Executando 3 fold cross-validation nos dados para achar o melhor conjunto de hiperparametros
    grid = GridSearchCV(modelo, hiperparametros, n_jobs = -1, cv = 3)
    grid.fit(data_treino, label_treino.values.ravel())
    # Salvando resultado do modelo para a etapa de refinamento
    primeira_etapa_refinamento = grid.best_params_
    # Pegando a vizinhanca do melhor valor
    hiperparametros =  vizinhanca_hiperparametros( primeira_etapa_refinamento, espalhamento)
    grid = GridSearchCV(modelo, hiperparametros, n_jobs = -1, cv = 3)
    grid.fit(data_treino, label_treino.values.ravel())
    return grid.best_params_

def vizinhanca_hiperparametros(resultados_modelos, espalhamento):
    hiperparametros_refinados = [0]*espalhamento
    espalhamento = int(espalhamento/2)
    # Para cada hiperparametro
    for j in range(len(list(resultados_modelos.keys()))):
        saltos = []
        valor_mediano = resultados_modelos[list(resultados_modelos.keys())[j]]
        for i in range(espalhamento):
            valor_mediano = int(valor_mediano/2)
            saltos.append(valor_mediano)
        valor_mediano = resultados_modelos[list(resultados_modelos.keys())[j]]
        # Criando o espalhamento dos valores do hiperparametro
        for k in range(len(hiperparametros_refinados)-1):
            if(k < len(saltos)):
                hiperparametros_refinados[k] = valor_mediano - saltos[k]
            else:
                hiperparametros_refinados[k] = valor_mediano + saltos[k-len(saltos)]
        hiperparametros_refinados[-1] = (valor_mediano)
        # Retirando possiveis valores repetidos e ordenando
        resultados_modelos[list(resultados_modelos.keys())[j]] = sorted(set(hiperparametros_refinados))
        
    return resultados_modelos  