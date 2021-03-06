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
from tratamentos import tratar_label
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split
from tratarDados import refinamento_hiperparametros

data_atual = date.today().strftime('%d/%m/%Y')
print(sys.argv)
# Primeiro argumento (1 para treinar o modelo e 0 para so executar)
if(len(sys.argv) == 1):
    TREINAR_MODELO = 0
else:
    TREINAR_MODELO = bool(int(sys.argv[1]))

if(TREINAR_MODELO):
    # Pega todos os dados do banco dremio
    data = todos_dados()
    # Carrega os dados validados
    dados_validados = pd.read_excel("dados_analisados.xlsx")
    # Retirando os dados validados do conjunto principal
    indexes = [0]*dados_validados.shape[0]
    for i in range(dados_validados.shape[0]):
        indexes[i] = data['Empenho (Sequencial Empenho)'][data['Empenho (Sequencial Empenho)'] == dados_validados['Empenho (Sequencial Empenho)(EOF)'].iloc[i]].index[0]
    data.drop(indexes,inplace = True)
    data.reset_index(drop = True, inplace = True)
    del indexes
    print(data.shape)
    # =============================================================================
    # Tratando os dados nao validados
    # =============================================================================
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
    del data_teste, label_teste
    label_treino.reset_index(drop = True, inplace = True)
    # Acha o melhor conjunto de hiperparametros para o algoritmo
    modelo = SVC(kernel="linear", random_state=0)
    hiperparametros = {'C':[0.1,1,10,50] }
    # Espalhamento se refere a quantidade de valores criados na vizinhanca do melhor valor para o fine search
    espalhamento = 3
    # Deixando pelo menos 3 documentos por classe para o teste
    label_treino, index_label_x_empenhos = tratar_label.label_elemento(label_treino["natureza_despesa_cod"], 3)
    data_treino =  pd.DataFrame.sparse.from_spmatrix(data_treino)
    data_treino.drop(index_label_x_empenhos,inplace = True, axis = 0)
    data_treino.reset_index(drop = True, inplace = True)
    data_treino = csr_matrix(data_treino)
    #
    melhor_c = refinamento_hiperparametros(data_treino, label_treino, modelo, hiperparametros, espalhamento)["C"]
    modelo = SVC(kernel="linear", random_state=0,C = melhor_c)
    # Treina o modelo de predicao da classe
    modelo.fit(data, label.values.ravel())
    with open('pickles/modelos_tratamentos/modelo_SVM.pk', 'wb') as fin:
        pickle.dump(modelo, fin)
    # =============================================================================
    # Tratando os dados validados 
    # =============================================================================
    label_validado = tratamentoDados(dados_validados.copy(),'dropar')
    dados_validados.drop("AN??LISE",inplace = True,axis = 1)
    tratamentoDados(dados_validados.copy(),'Modelo 2 tfidf')
    tratamentoDados(dados_validados.copy(),'Modelo 2 OHE')
    data_validado = pickles.carregarPickle("data_validado")
    tfidf_validado = pickles.carregarPickle("tfidf_validado")
    data_validado = sparse.hstack((csr_matrix(data_validado),csr_matrix(tfidf_validado) ))
    del tfidf_validado, dados_validados
    print(data_validado.shape)
    # =============================================================================
    # Treino do modelo de predicao de corretude    
    # =============================================================================
    # Separando 40% dos dados para treino e 40% para teste
    data_treino, data_teste, label_treino, label_teste = train_test_split(data_validado, label_validado, test_size = 0.6,stratify = label_validado, random_state = 10)
    del data_teste, label_teste
    label_treino.reset_index(drop = True, inplace = True)
    # Acha o melhor conjunto de hiperparametros para o algoritmo
    modelo = SVC(kernel="linear", random_state=0)
    hiperparametros = {'C':[0.1,1,10,50] }
    espalhamento = 3
    melhor_c = refinamento_hiperparametros(data_treino, label_treino, modelo, hiperparametros, espalhamento)["C"]
    modelo_validado = SVC(kernel="linear", random_state=0,C = melhor_c)
    modelo_validado.fit(data_validado, label_validado.values.ravel())
    with open('pickles/modelos_tratamentos/modelo_SVM_validado.pk', 'wb') as fin:
        pickle.dump(modelo_validado, fin)
    # Salvando o rotulo validado
    pickles.criarPickle(label_validado,"label_validado")
    
else:
    label = pickles.carregarPickle("label")
    label_validado = pickles.carregarPickle("label_validado")
    with open('pickles/modelos_tratamentos/modelo_SVM.pk', 'rb') as pickle_file:
        modelo = pickle.load(pickle_file)
    with open('pickles/modelos_tratamentos/modelo_SVM_validado.pk', 'rb') as pickle_file:
        modelo_validado = pickle.load(pickle_file)
    # Pega os novos dados
    if(len(sys.argv) == 1):
        dados_novos = range_dados()
    elif(len(sys.argv) == 3):
        dados_novos = range_dados(sys.argv[2])
    else:
        dados_novos = range_dados(sys.argv[2],sys.argv[3])
    if(dados_novos.shape[0] == 0):
        print("0 Documentos encontrados")
        sys.exit()
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
    # Verificando se os dados novos estao dentro dos dados de treino
    if(dados_novos.shape[0] != 0):
        # Trata os dados
        dados_hoje_ohe, label_hoje = tratarDados(dados_novos,"OHE")
        dados_hoje_tfidf, label_hoje = tratarDados(dados_novos,"tfidf")
        dados_hoje = sparse.hstack((csr_matrix(dados_hoje_ohe),csr_matrix(dados_hoje_tfidf) ))
        del dados_hoje_ohe,dados_hoje_tfidf
        identificador_empenho = pickles.carregarPickle("modelos_tratamentos/identificador_empenho")
        y_predito = modelo.predict(dados_hoje)
        # Tratar os dados para segundo modelo
        dados_hoje_ohe, label_hoje = tratarDados(dados_novos,"Modelo 2 OHE")
        dados_hoje_tfidf, label_hoje = tratarDados(dados_novos,"Modelo 2 tfidf")
        dados_hoje = sparse.hstack((csr_matrix(dados_hoje_ohe),csr_matrix(dados_hoje_tfidf) ))
        y_predito_validado = modelo_validado.predict(dados_hoje)
        # Metricas de avaliacao
        micro = f1_score(label_hoje,y_predito,average='micro')
        macro = f1_score(label_hoje,y_predito,average='macro')
        string = ""
        valor_c = modelo.C
        print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"??: ",micro)
        print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"??: ",macro)
        resultado = pd.concat([pd.DataFrame(identificador_empenho),pd.DataFrame(label_hoje)],axis = 1)
        resultado = pd.concat([resultado,pd.DataFrame(y_predito)],axis = 1)
        resultado = pd.concat([resultado,pd.DataFrame(y_predito_validado)],axis = 1)
        colunas = ['empenho_sequencial_empenho','natureza_real','natureza_predita',"corretude"]
        resultado.columns = colunas
    else:
        dados_hoje = dados_novos
        label_hoje = y_predito = y_predito_validado = micro = macro = pd.DataFrame()
        colunas = ['empenho_sequencial_empenho','natureza_real','natureza_predita',"corretude"]
        resultado = pd.DataFrame(columns=colunas)
        
    if(len(dados_fora_modelo) >0):
        label_inconclusiva = ["inconclusivo"]*len(dados_fora_modelo)
        identificador_empenho_inconclusivo = pd.DataFrame(dados_fora_modelo['Empenho (Sequencial Empenho)'])
        resultado_inconclusivo = pd.concat([pd.DataFrame(identificador_empenho_inconclusivo),dados_fora_modelo['Natureza Despesa (Cod)']],axis = 1)
        resultado_inconclusivo = pd.concat([resultado_inconclusivo,pd.DataFrame(label_inconclusiva)],axis = 1)
        resultado_inconclusivo.columns = colunas[:-1]
        # Junta os resultados
        resultado = pd.concat([resultado,resultado_inconclusivo],axis = 0)
        resultado["corretude"].fillna("INCONCLUSIVO", inplace = True)
        del resultado_inconclusivo
    resultado['data_predicao'] = data_atual
    resultado.reset_index(inplace = True,drop=True)
    resultado['Resultado'] = resultado['natureza_real']==resultado['natureza_predita']
    conclusao = [""]*resultado.shape[0]
    # Faz a logica das saidas
    for l in range(resultado.shape[0]):
        if(resultado['Resultado'] .iloc[l] and resultado['corretude'].iloc[l] == "OK" ):
            conclusao[l] = "C_M1-M2"
        elif(resultado['Resultado'] .iloc[l] and resultado['corretude'].iloc[l] == "INCORRETO"):
            conclusao[l] = "INCV_M2"
        elif(resultado['Resultado'] .iloc[l] and resultado['corretude'].iloc[l] == "INCONCLUSIVO"):
            conclusao[l] = "AD_M2"
        elif(resultado['Resultado'] .iloc[l] == False and resultado['corretude'].iloc[l] == "OK"):
            conclusao[l] = "INCV_M1"
        elif(resultado['Resultado'] .iloc[l] == False and resultado['corretude'].iloc[l] == "INCORRETO"):
            conclusao[l] = "INCT_M1-M2"
        elif(resultado['Resultado'] .iloc[l] == False and resultado['corretude'].iloc[l] == "INCONCLUSIVO"):
            conclusao[l] = "INCV_M1-AD_M2"
    resultado['Resultado'] = conclusao
    if(len(sys.argv) == 1):
        resultado.to_csv("resultado-"+(date.today()-timedelta(days=1)).strftime('%Y/%m/%d').replace("/","-")+".csv",index = False)
    else:
        resultado.to_csv("resultado-"+sys.argv[2].replace("/","-")+".csv",index = False)
    # =============================================================================
    # LOG
    # =============================================================================
    f = open('log.txt','a+')
    f.write(data_atual+'\n')
    f.write("Quantidade total de documentos : "+str(len(resultado))+'\n')
    f.write("Quantidade de documentos preditos: "+str(label_hoje.shape[0])+'\n')
    f.write("Quantidade de incorretos: "+str(len(resultado['Resultado'][resultado['Resultado'] == "INCT_M1-M2"]))+'\n')
    f.write("Quantidade de corretos: "+str(len(resultado['Resultado'][resultado['Resultado'] == "C_M1-M2"]))+'\n')
    f.write("Micro de acerto: "+str(micro *100)[:5]+"%"+"\n")
    f.write("Macro de acerto: "+str(macro *100)[:5]+"%"+"\n")
    f.write("="*50+"\n")
    f.flush()
    f.close()