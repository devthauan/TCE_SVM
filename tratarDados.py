### Bibliotecas python ###
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing
from scipy.sparse import csr_matrix
### Meus pacotes ###
from tratamentos import pickles
from tratamentos import tratar_texto
from tratamentos import one_hot_encoding
### Meus pacotes ###

def tratarDados(data):
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    identificador_empenho = pd.DataFrame(data['empenho_sequencial_empenho'])
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome','valor_estorno_anulacao_empenho',
                      'valor_anulacao_cancelamento_empenho','fonte_recurso_cod',
                      'elemento_despesa','grupo_despesa','empenho_sequencial_empenho','periodo'], axis='columns')
    # rotulo
    label = data['natureza_despesa_cod']
    label = pd.DataFrame(label)
    data = data.drop('natureza_despesa_cod',axis = 1)
    # tfidf
    textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
    # Função que gera o TF-IDF do texto tratado
    with open('pickles/modelos_tratamentos/tfidf_modelo.pk', 'rb') as pickle_file:
        tfidf_modelo = pickle.load(pickle_file)
    tfidf =  pd.DataFrame.sparse.from_spmatrix(tfidf_modelo.transform(textoTratado))
    del textoTratado
    data = data.drop('empenho_historico',axis = 1)
# =============================================================================
#     Tratamento dos dados
# =============================================================================
    # Codigo que gera o meta atributo "pessoa_juridica" onde 1 representa que a pessoa e juridica e 0 caso seja fisica
    identificacao_pessoa = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(data['beneficiario_cpf'].iloc[i] == "-" or data['beneficiario_cpf'].iloc[i] is None or np.isnan(data['beneficiario_cpf'].iloc[i])):
        identificacao_pessoa[i] = 1
      else: identificacao_pessoa[i]=0
    data['pessoa_juridica'] = identificacao_pessoa
    del identificacao_pessoa
    data['pessoa_juridica'] = data['pessoa_juridica'].astype("int8")
    data = data.drop(["beneficiario_cpf","beneficiario_cnpj","beneficiario_cpf/cnpj"], axis='columns')
    # Tratando o campo beneficiario nome como texto livre e fazendo TFIDF
    texto_beneficiario = tratar_texto.cleanTextData(data["beneficiario_nome"])
    with open('pickles/modelos_tratamentos/tfidf_beneficiario.pk', 'rb') as pickle_file:
                tfidf_beneficiario_modelo = pickle.load(pickle_file)
    tfidf_beneficiario = pd.DataFrame.sparse.from_spmatrix(tfidf_beneficiario_modelo.transform(texto_beneficiario))
    del texto_beneficiario
    data = data.drop("beneficiario_nome", axis='columns')
    
    # Codigo que gera o meta atributo "orgao_sucedido" onde 1 representa que o orgao tem um novo orgao sucessor e 0 caso contrario
    orgao_sucedido = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(data['orgao'].iloc[i] != data['orgao_sucessor_atual'].iloc[i]):
        orgao_sucedido[i] = 1
      else:
        orgao_sucedido[i] = 0
    data['orgao_sucedido'] = orgao_sucedido
    del orgao_sucedido
    data['orgao_sucedido'] = data['orgao_sucedido'].astype("int8")
    data = data.drop(["orgao"], axis='columns')
    
    # Codigo que retira o codigo de programa (retirando 10 valores)
    nome = [0] * data.shape[0]
    for i in range(len(data['programa'])):
      nome[i] = (data['programa'].iloc[i][7:])
    data['programa'] = nome
    del nome
    
    # Codigo que retira o codigo de acao (retirando 77 valores)
    nome = [0] * data.shape[0]
    for i in range(len(data['acao'])):
      nome[i] = (data['acao'].iloc[i][7:])
    data['acao'] = nome
    del nome
    
    # Codigo que concatena acao e programa
    acao_programa = [0] * data.shape[0]
    for i in range(data.shape[0]):
      acao_programa[i] = (data['acao'].iloc[i] + " & " + data['programa'].iloc[i])
    data['acao_programa'] = acao_programa
    del acao_programa
    data = data.drop(["acao","programa"],axis = 1)   
    
    # Codigo que mostra a quantidade de empenhos por processo
    quantidade_empenhos_processo = data['empenho_numero_do_processo'].value_counts()
    quantidade_empenhos_processo = quantidade_empenhos_processo.to_dict()
    empenhos_processo = [0]* data.shape[0]
    for i in range(data.shape[0]):
        empenhos_processo[i] = quantidade_empenhos_processo[data['empenho_numero_do_processo'].iloc[i]]
    data['empenhos_por_processo'] = empenhos_processo
    del empenhos_processo
    del quantidade_empenhos_processo
    data = data.drop('empenho_numero_do_processo',axis = 1)
# =============================================================================
#     Tratamento dos dados
# =============================================================================
    # Normalizando colunas numéricas
    colunas = data.columns
    for col in colunas:
        if(data[col].dtype != "O"):
            with open('pickles/modelos_tratamentos/'+"normalization_"+col+'.pk', 'rb') as pickle_file:
                min_max_scaler = pickle.load(pickle_file)
            data[col] = pd.DataFrame(min_max_scaler.transform(data[col].values.reshape(-1,1)))
    # OHE
    data = one_hot_encoding.aplyOHE(data)
    # Excluindo as colunas que ja foram tratadas
    data = pd.concat([data,tfidf_beneficiario],axis = 1)
    aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
    data =  pd.DataFrame.sparse.from_spmatrix(aux)
    pickles.criaPickle(identificador_empenho,"modelos_tratamentos/identificador_empenho")
    return data, label