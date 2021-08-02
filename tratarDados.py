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

def tratarDados(data, opcao = "visao dupla"):
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
    if("Modelo 2" not in opcao):
        data = data.drop('natureza_despesa_cod',axis = 1)
    # tfidf
    textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
    if("Modelo 2" in opcao):
        # Função que gera o TF-IDF do texto tratado
        with open('pickles/modelos_tratamentos/tfidf_modelo_validado'+'.pk', 'rb') as pickle_file:
            tfidf_modelo = pickle.load(pickle_file)
    else:
        # Função que gera o TF-IDF do texto tratado
        with open('pickles/modelos_tratamentos/tfidf_modelo'+'.pk', 'rb') as pickle_file:
            tfidf_modelo = pickle.load(pickle_file)
    tfidf =  pd.DataFrame.sparse.from_spmatrix(tfidf_modelo.transform(textoTratado))
    del textoTratado
    data = data.drop('empenho_historico',axis = 1)
    # Tratamento dos dados
    data = tratamento_especifico(data.copy())
    # Tratando o beneficiario nome
    nome = [""]*data.shape[0]
    for i in range(data.shape[0]):
        if(data['pessoa_juridica'].iloc[i]):
            nome[i] = data["beneficiario_nome"].iloc[i]
        else:
            nome[i] = "pessoafisica"
    data["beneficiario_nome"] = nome
    # Tratando o campo beneficiario nome como texto livre e fazendo TFIDF
    texto_beneficiario = tratar_texto.cleanTextData(data["beneficiario_nome"])
    if("Modelo 2" in opcao):
        with open('pickles/modelos_tratamentos/tfidf_beneficiario_validado'+'.pk', 'rb') as pickle_file:
            tfidf_beneficiario = pickle.load(pickle_file)
    else:
        with open('pickles/modelos_tratamentos/tfidf_beneficiario'+'.pk', 'rb') as pickle_file:
            tfidf_beneficiario = pickle.load(pickle_file)
    data_cv = tfidf_beneficiario.transform(texto_beneficiario)
    tfidf_beneficiario = pd.DataFrame.sparse.from_spmatrix(data_cv, columns = tfidf_beneficiario.get_feature_names())
    data = data.drop("beneficiario_nome", axis='columns')
    if("Modelo 2" in opcao):
        sufixo = "_validado"
        pickles.criarPickle(tfidf_beneficiario,"dados_tfidf_beneficiario_validado")
    else:
        sufixo = ""
        pickles.criarPickle(tfidf_beneficiario,"dados_tfidf_beneficiario")
    # Normalizando colunas numéricas
    colunas = data.columns
    for col in colunas:
        if(data[col].dtype != "O"):
            with open('pickles/modelos_tratamentos/'+"normalization_"+col+sufixo+'.pk', 'rb') as pickle_file:
                min_max_scaler = pickle.load(pickle_file)
            data[col] = pd.DataFrame(min_max_scaler.transform(data[col].values.reshape(-1,1)))
    # OHE
    if("OHE" in opcao):
        data = one_hot_encoding.aplyOHE(data, opcao)
        if("Modelo 2" in opcao):
            tfidf_beneficiario = pickles.carregarPickle("dados_tfidf_beneficiario_validado")
        else:
            tfidf_beneficiario = pickles.carregarPickle("dados_tfidf_beneficiario")
        data = pd.concat([data, tfidf_beneficiario], axis = 1)
        return data, label
    elif("tfidf" in opcao):
        return tfidf, label
    else:
        data = one_hot_encoding.aplyOHE(data)
        if("Modelo 2" in opcao):
            tfidf_beneficiario = pickles.carregarPickle("dados_tfidf_beneficiario_validado")
        else:
            tfidf_beneficiario = pickles.carregarPickle("dados_tfidf_beneficiario")
        data = pd.concat([data, tfidf_beneficiario], axis = 1)
        data = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
        data =  pd.DataFrame.sparse.from_spmatrix(data)
        return data, label


def refinamento_hiperparametros(data_treino, label_treino, modelo , hiperparametros, espalhamento):
    # Executando 3 fold cross-validation nos dados para achar o melhor conjunto de hiperparametros
    grid = GridSearchCV(modelo, hiperparametros, n_jobs = -1, cv = 3)
    grid.fit(data_treino, label_treino.values.ravel())
    # Salvando resultado do modelo para a etapa de refinamento
    primeira_etapa_refinamento = grid.best_params_
    # Pegando a vizinhanca do melhor valor
    hiperparametros =  vizinhanca_hiperparametros( primeira_etapa_refinamento, espalhamento, hiperparametros)
    grid = GridSearchCV(modelo, hiperparametros, n_jobs = -1, cv = 3)
    grid.fit(data_treino, label_treino.values.ravel())
    return grid.best_params_

def vizinhanca_hiperparametros(resultados_modelos, espalhamento, hiperparametros):
    hiperparametros_refinados = [0]*espalhamento
    espalhamento = int(espalhamento/2)
    # Para cada hiperparametro
    for j in range(len(list(resultados_modelos.keys()))):
        saltos = []
        # Calculo de distancia do ponto mais longe ao central sera um decimo da diferenca dos saltos
        distancia = 2* int((hiperparametros[list(hiperparametros.keys())[j]][1] - hiperparametros[list(hiperparametros.keys())[j]][0])/10)
        for i in range(espalhamento):
            distancia = int(distancia/2)
            saltos.append(distancia)
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