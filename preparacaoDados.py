### Bibliotecas python ###
import pickle
import pandas as pd
from tratamentos import pickles
from sklearn import preprocessing
### Meus pacotes ###
from tratamentos import tratar_label
from tratamentos import tratar_texto
from tratamentos import one_hot_encoding
### Meus pacotes ###

def tratamentoDados(data, escolha):
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    data = filtro(data.copy())
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['exercicio_do_orcamento_ano','classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome',
                      'valor_estorno_anulacao_empenho','valor_anulacao_cancelamento_empenho',
                      'fonte_recurso_cod','elemento_despesa','grupo_despesa',
                      'empenho_sequencial_empenho'], axis='columns')
    # Funcao que separa o rotulo do dataset e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
    label,linhas_label_unica = tratar_label.tratarLabel(data)
    label = pd.DataFrame(label)
    # Excluindo as naturezas de despesas que so tem 1 empenho do dataset
    data = data.drop(linhas_label_unica, axis = 0)
    data.reset_index(drop = True, inplace = True)
    del linhas_label_unica
    
    if(escolha == "dropar"):
        return data["analise"]         
    if(escolha == "tfidf"):
        # Funcao que limpa o texto retira stopwords acentos pontuacao etc.
        textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
        # Função que gera o TF-IDF do texto tratado
        tfidf = tratar_texto.calculaTFIDF(textoTratado)
        del textoTratado
        pickles.criarPickle(tfidf,'tfidf')
        return 0
    # Tratamento dos dados
    data = tratamento_especifico(data.copy())
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
    if(escolha == "OHE"):
        # Aplicando a estrategia One Hot Encoding
        data = one_hot_encoding.oneHotEncoding(data)
        #data = pd.concat([data,tfidf_beneficiario],axis = 1)
        pickles.criarPickle(data,'data')
        pickles.criarPickle(label,'label')
    else: return None


def tratamento_especifico(data):
    # Trata o campo periodo pegando apenas o mes
    periodo_mes = [0]*len(data["periodo"])
    for i in range(len(periodo_mes)):
        periodo_mes[i] =  str(data["periodo"].iloc[i])[5:7]
    data["periodo"] = pd.DataFrame(periodo_mes)
    # Codigo que gera o meta-atributo "pessoa_juridica" onde 1 representa que a pessoa e juridica e 0 caso seja fisica
    identificacao_pessoa = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(len(str(data['beneficiario_cpf'].iloc[i])) >1 ):
        identificacao_pessoa[i] = 1
      else: identificacao_pessoa[i]=0
    data['pessoa_juridica'] = identificacao_pessoa
    data['pessoa_juridica'] = data['pessoa_juridica'].astype("int8")
    del identificacao_pessoa
    data = data.drop(["beneficiario_cpf","beneficiario_cnpj","beneficiario_cpf/cnpj",
                      "beneficiario_nome"], axis='columns')
    # Codigo que gera o meta atributo "orgao_sucedido" onde 1 representa que o orgao tem um novo orgao sucessor e 0 caso contrario
    orgao_sucedido = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(data['orgao'].iloc[i] != data['orgao_sucessor_atual'].iloc[i]):
        orgao_sucedido[i] = 1
      else:
        orgao_sucedido[i] = 0
    data['orgao_sucedido'] = orgao_sucedido
    data['orgao_sucedido'] = data['orgao_sucedido'].astype("int8")
    del orgao_sucedido
    data = data.drop(["orgao"], axis='columns')
    
    # Codigo que retira o codigo de programa e acao
    nome_programa = [0] * data.shape[0]
    nome_acao = [0] * data.shape[0]
    for i in range(len(data['programa'])):
      nome_programa[i] = (data['programa'].iloc[i][7:])
      nome_acao[i] = (data['acao'].iloc[i][7:])
    data['programa'] = nome_programa
    data['acao'] = nome_acao
    del nome_programa, nome_acao
    
    # Codigo que junta acao e programa
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
    del empenhos_processo, quantidade_empenhos_processo
    data = data.drop('empenho_numero_do_processo',axis = 1)
    return data

def filtro(data):
    # Excluindo empenhos pertencentes ao elemento de despesa 92
    exercicio_anterior = data['natureza_despesa_cod'].str.contains(".\..\...\.92\...", regex= True, na=False)
    index = exercicio_anterior.where(exercicio_anterior==True).dropna().index
    data.drop(index, inplace = True)
    data.reset_index(drop=True, inplace=True)
    del exercicio_anterior
    # Deletando empenhos sem relevancia devido ao saldo zerado
    index = data["valor_saldo_do_empenho"].where(data["valor_saldo_do_empenho"] == 0).dropna().index
    data.drop(index,inplace = True)
    data.reset_index(drop=True, inplace=True)
    # Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
    label = data['natureza_despesa_cod']
    sem_relevancia =pd.DataFrame( pd.read_excel("Naturezas de despesa com vigência encerrada.xlsx")['Nat. Despesa'] )
    excluir = []
    for i in range(len(sem_relevancia['Nat. Despesa'])):
        excluir.append( label[label == sem_relevancia['Nat. Despesa'].iloc[i]].index )
    excluir = [item for sublist in excluir for item in sublist]
    # Excluindo as naturezas que nao estao mais vigentes
    data.drop(excluir,inplace = True)
    data.reset_index(drop=True, inplace=True)
    return data