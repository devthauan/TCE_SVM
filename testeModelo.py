# -*- coding: utf-8 -*-
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import date
from sklearn.svm import SVC
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from conexaoDados import range_dados
from preparacaoDados import tratamentoDados

data_atual = date.today().strftime('%d/%m/%Y')
print(sys.argv)
# Primeiro argumento (1 para treinar o modelo e 0 para so executar)
if(len(sys.argv) == 1):
    TREINAR_MODELO = 0
else:
    TREINAR_MODELO = bool(int(sys.argv[1]))

if(TREINAR_MODELO):
    tratamentoDados('tfidf')
    tratamentoDados('OHE')
    data = pickles.carregaPickle("data")
    label = pickles.carregaPickle("label")
    tfidf = pickles.carregaPickle("tfidf")
    data = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
    del tfidf
    print(data.shape)
    #Treinar modelo
    modelo = SVC(kernel="linear",C= 10,random_state=0)
    modelo.fit(data, label.values.ravel())
    with open('pickles/modelos_tratamentos/modelo_SVM.pk', 'wb') as fin:
        pickle.dump(modelo, fin)
else:
    label = pickles.carregaPickle("label")
    with open('pickles/modelos_tratamentos/modelo_SVM.pk', 'rb') as pickle_file:
        modelo = pickle.load(pickle_file)
    # Pega os novos dados
    if(len(sys.argv) == 1):
        dados_novos = range_dados()
    else:
        dados_novos = range_dados(sys.argv[2])
    dados_novos.reset_index(inplace = True,drop=True)
    naturezas_novas = pd.DataFrame(dados_novos['Natureza Despesa (Cod)'])
    fora_do_modelo = []
    label_classes = list(label['natureza_despesa_cod'].value_counts().index)
    for i in range(len(naturezas_novas)):
        # Verifica se os novos dados estao presentes no modelo, caso nao estejam adiciona-os em um vetor separado
        if(naturezas_novas.iloc[i][0] not in label_classes):
            fora_do_modelo.append(i)
    del naturezas_novas
    dados_fora_modelo = dados_novos.iloc[fora_do_modelo]
    dados_novos.drop(fora_do_modelo,inplace = True)
    dados_novos.reset_index(inplace = True,drop=True)
    dados_fora_modelo.reset_index(inplace = True,drop=True)
    # Trata os dados
    dados_hoje, label_hoje = tratarDados(dados_novos)
    del dados_novos
    identificador_empenho = pickles.carregaPickle("modelos_tratamentos/identificador_empenho")
    y_predito = modelo.predict(dados_hoje)
    micro = f1_score(label_hoje,y_predito,average='micro')
    macro = f1_score(label_hoje,y_predito,average='macro')
    string = ""
    valor_c = 10
    print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"é: ",micro)
    print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"é: ",macro)
    resultado = pd.concat([pd.DataFrame(identificador_empenho),pd.DataFrame(label_hoje)],axis = 1)
    resultado = pd.concat([resultado,pd.DataFrame(y_predito)],axis = 1)
    colunas = ['empenho_sequencial_empenho','natureza_real','natureza_predita']
    resultado.columns = colunas
    #
    label_inconclusiva = ["inconclusivo"]*len(dados_fora_modelo)
    identificador_empenho_inconclusivo = pd.DataFrame(dados_fora_modelo['Empenho (Sequencial Empenho)'])
    resultado_inconclusivo = pd.concat([pd.DataFrame(identificador_empenho_inconclusivo),dados_fora_modelo['Natureza Despesa (Cod)']],axis = 1)
    resultado_inconclusivo = pd.concat([resultado_inconclusivo,pd.DataFrame(label_inconclusiva)],axis = 1)
    resultado_inconclusivo.columns = colunas
    # Junta os resultados
    resultado = pd.concat([resultado,resultado_inconclusivo],axis = 0)
    del resultado_inconclusivo
    resultado['data_predicao'] = data_atual
    resultado.reset_index(inplace = True,drop=True)
    resultado['acerto'] = resultado['natureza_real']==resultado['natureza_predita']
    resultado.to_csv("resultado.csv",index = False)
    # =============================================================================
    # LOG
    # =============================================================================
    f = open('log.txt','a+')
    f.write(data_atual+'\n')
    f.write("Quantidade total de documentos : "+str(len(resultado))+'\n')
    f.write("Quantidade de documentos preditos: "+str(label_hoje.shape[0])+'\n')
    f.write("Quantidade de acertos: "+str(len(resultado['acerto'][resultado['acerto'] == True]))+'\n')
    f.write("Quantidade de erros: "+str(len(resultado['acerto'][resultado['acerto'] == False]) - len(resultado['natureza_predita'][resultado['natureza_predita']=='inconclusivo']))+'\n')
    f.write("Quantidade de inconclusivos: "+str(len(resultado['natureza_predita'][resultado['natureza_predita']=='inconclusivo']))+'\n')
    f.write("Micro de acerto: "+str(micro *100)[:5]+"%"+"\n")
    f.write("Macro de acerto: "+str(macro *100)[:5]+"%"+"\n")
    f.write("="*50+"\n")
    f.flush()
    f.close()