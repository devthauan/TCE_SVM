### Bibliotecas python ###
import pickle
import numpy as np
import pandas as pd
from tratamentos import pickles
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
### Meus pacotes ###
from tratamentos import tratar_label
from tratamentos import tratar_texto
from tratamentos import one_hot_encoding
### Meus pacotes ###

def tratamentoDados(escolha):
    # Pega todos os dados do banco dremio
    from conexaoDados import todos_dados
    data = todos_dados()
    # Carrega os dados na variavel 'data' utilizando o Pandas
#    data = pickles.carregaPickle("df")
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    # Excluindo empenhos diferentes aglomerados na classe 92
    exercicio_anterior = data['natureza_despesa_cod'].str.contains(".\..\...\.92\...", regex= True, na=False)
    index = exercicio_anterior.where(exercicio_anterior==True).dropna().index
    data.drop(index,inplace = True)
    data.reset_index(drop=True, inplace=True)
    del exercicio_anterior
    # Deletando empenhos sem relevancia devido ao saldo zerado
    index = data["valor_saldo_do_empenho"].where(data["valor_saldo_do_empenho"] == 0).dropna().index
    data.drop(index,inplace = True)
    data.reset_index(drop=True, inplace=True)
#    data = data[:10000] #limitando os dados para fazer testes
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome','valor_estorno_anulacao_empenho',
                      'valor_anulacao_cancelamento_empenho','fonte_recurso_cod',
                      'elemento_despesa','grupo_despesa','empenho_sequencial_empenho','periodo'], axis='columns')
    # Funcao que gera o rotulo e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
    label,linhas_label_unica = tratar_label.tratarLabel(data)
    label = pd.DataFrame(label)
    # Excluindo as naturezas de despesas que so tem 1 empenho
    data = data.drop(linhas_label_unica)
    data.reset_index(drop=True, inplace=True)
    del linhas_label_unica
    # Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
    sem_relevancia = pd.read_excel("Naturezas de despesa com vigência encerrada.xlsx")
    sem_relevancia = sem_relevancia['Nat. Despesa']
    sem_relevancia = pd.DataFrame(sem_relevancia)
    excluir = []
    for i in range(len(sem_relevancia['Nat. Despesa'])):
        excluir.append( label.where( label['natureza_despesa_cod'] == sem_relevancia['Nat. Despesa'].iloc[i] ).dropna().index )
    excluir = [item for sublist in excluir for item in sublist]
    # Excluindo as naturezas que nao estao mais vigentes
    label.drop(excluir,inplace =True)
    label.reset_index(drop=True, inplace=True)
    data.drop(excluir,inplace = True)
    data.reset_index(drop=True, inplace=True)
    del excluir, sem_relevancia
    #Codigo para pegar 60% dos dados estratificado 
# =============================================================================
#     #Pegando 60% stratificado para teste
# =============================================================================
#    X_train, data, y_train, label = train_test_split(data, label,test_size=0.6,stratify = label,random_state =5)
#    del X_train,y_train
#    data.reset_index(drop=True,inplace=True)
#    label.reset_index(drop=True,inplace=True)
#    # Pegando agora as classes com 1 natureza apenas para excluir
#    label,excluir = tratar_label.label_1_elmento(label)
#    data.drop(excluir,inplace=True)
#    data.reset_index(drop=True,inplace=True)
#    del excluir
# =============================================================================
#     #Pegando 60% stratificado para teste
# =============================================================================
    if(escolha == "tfidf"):
        # Funcao que limpa o texto retira stopwords acentos pontuacao etc.
        textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
        # Função que gera o TF-IDF do texto tratado
        tfidf = tratar_texto.calculaTFIDF(textoTratado)
        del textoTratado
#        return tfidf
        pickles.criaPickle(tfidf,'tfidf')
    if(escolha == "texto"):
         # Funcao que limpa o texto retira stopwords acentos pontuacao etc.
        textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
#        return textoTratado
        pickles.criaPickle(pd.DataFrame(textoTratado),'textoTratado')
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
    cv = TfidfVectorizer(dtype=np.float32)
    data_cv = cv.fit(texto_beneficiario)
    with open('pickles/modelos_tratamentos/tfidf_beneficiario.pk', 'wb') as fin:
        pickle.dump(cv, fin)
    data_cv = cv.transform(texto_beneficiario)
    tfidf_beneficiario = pd.DataFrame.sparse.from_spmatrix(data_cv, columns = cv.get_feature_names())
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
    min_max_scaler = preprocessing.MinMaxScaler()
    colunas = data.columns
    for col in colunas:
        if(data[col].dtype != "O"):
            min_max_scaler.fit(data[col].values.reshape(-1,1))
            with open('pickles/modelos_tratamentos/'+"normalization_"+col+'.pk', 'wb') as fin:
                pickle.dump(min_max_scaler, fin)
            data[col] = pd.DataFrame(min_max_scaler.transform(data[col].values.reshape(-1,1)))
            
    # Excluindo as colunas que ja foram tratadas
    data = data.drop(['empenho_historico','natureza_despesa_cod'], axis='columns')
    if(escolha == "sem OHE"):
        data = pd.concat([data,tfidf_beneficiario],axis = 1)
        return data, label
    elif(escolha == "OHE"):
        # Aplicando a estrategia One Hot Encoding
        data = one_hot_encoding.oneHotEncoding(data)
        data = pd.concat([data,tfidf_beneficiario],axis = 1)
#        return data, label
        pickles.criaPickle(data,'data')
        pickles.criaPickle(label,'label')
    else: return None
###########################DADOS TRTADOS######################################