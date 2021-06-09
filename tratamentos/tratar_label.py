import pandas as pd
def tratarLabel(data):
    label = data['natureza_despesa_cod']
    # Pegando os rotulos e a quantidade de empenhos pertencentes a eles
    quantidade_labels = pd.DataFrame(label.value_counts(ascending = True))
    # Pegando o nome dos rotulos que so aparecem x vezes
    label_poucos_documentos = []
    for i in range(quantidade_labels.shape[0]):
        if(quantidade_labels.iloc[i].values[0] <= 6):
            label_poucos_documentos.append(quantidade_labels.iloc[i].name)
        else:
            break
    # Pegando as linhas das classes com só x documento
    index_label_poucos_documento = []
    for i in range(len(label_poucos_documentos)):
        index_label_poucos_documento.append(data[data["natureza_despesa_cod"] == label_poucos_documentos[i]].index)   
    # Achatando o vetor de vetores
    index_label_poucos_documento = [item for sublist in index_label_poucos_documento for item in sublist]
    # Excluindo
    label = label.drop(index_label_poucos_documento, axis = 0)
    label.reset_index(drop=True, inplace=True)
    return label, index_label_poucos_documento

def label_elemento(label, quantidade):
    quantidade_labels = pd.DataFrame(label.value_counts(ascending = True))
    #pegando o nome das labels que aparecem a quantidade de vezes
    label_x_documento_apenas = []
    for i in range(quantidade_labels.shape[0]):
        if(quantidade_labels.iloc[i].values <= quantidade):
            label_x_documento_apenas.append(quantidade_labels.iloc[i].name)
        else:
            break
    #pegando as linhas das classes com menos que a quantidade indicada
    index_label_unico_documento = []
    for i in range(label.shape[0]):
        if(label.iloc[i] in label_x_documento_apenas):
            index_label_unico_documento.append(i)
    
    label = label.drop(index_label_unico_documento)
    label.reset_index(drop=True, inplace=True)
    return label, index_label_unico_documento